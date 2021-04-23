import json
import os
import math
import glob
import numpy as np
import png
import shutil
from src.utility.LockerUtility import Locker

import bpy
from mathutils import Euler, Matrix, Vector

from src.utility.BlenderUtility import get_all_blender_mesh_objects, load_image
from src.utility.Utility import Utility
from src.utility import TransformUtility as T
from src.writer.WriterInterface import WriterInterface
from src.writer.CameraStateWriter import CameraStateWriter
from src.main.GlobalStorage import GlobalStorage
import cv2
from PIL import Image


def load_json(path, keys_to_int=False):
    """Loads content of a JSON file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the JSON file.
    :return: Content of the loaded JSON file.
    """

    # Keys to integers.
    def convert_keys_to_int(x):
        return {int(k) if k.lstrip('-').isdigit() else k: v for k, v in x.items()}

    with open(path, 'r') as f:
        if keys_to_int:
            content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
        else:
            content = json.load(f)

    return content


def save_json(path, content):
    """ Saves the content to a JSON file in a human-friendly format.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    """
    with open(path, 'w') as f:

        if isinstance(content, dict):
            f.write('{\n')
            content_sorted = sorted(content.items(), key=lambda x: x[0])
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write(
                    '  \"{}\": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(',')
                f.write('\n')
            f.write('}')

        elif isinstance(content, list):
            f.write('[\n')
            for elem_id, elem in enumerate(content):
                f.write('  {}'.format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(',')
                f.write('\n')
            f.write(']')

        else:
            json.dump(content, f, sort_keys=True)


def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im[im > 65535] = 65535
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def save_segmap(path, im):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    if(im.dtype =='uint8'):
        bits = 8
    elif(im.dtype=='uint16'):
        bits = 16
    elif(im.dtype=='uint32'):
        raise ValueError('Saving 32-bit images not supported ')
    else:
        raise ValueError('Segmap needs to be uint 8 or 16')

    if(len(im.shape)>2):
        raise ValueError('Only supports single channel segmaps')


    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    # w_segmap = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    # with open(path, 'wb') as f:
    #     w_segmap.write(f, np.reshape(im, (-1, im.shape[1])))

    cv2.imwrite(path,im)


class MyWriter(WriterInterface):
    """ Saves the synthesized dataset in the BOP format. The dataset is split
        into chunks which are saved as individual "scenes". For more details
        about the BOP format, visit the BOP toolkit docs:
        https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md

    **Attributes per object**:

    .. list-table::
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - dataset
          - Only save annotations for objects of the specified bop dataset. Saves all object poses if not defined.
            Default: ''
          - string
        * - append_to_existing_output
          - If true, the new frames will be appended to the existing ones. Default: False
          - bool
        * - save_world2cam
          - If true, camera to world transformations "cam_R_w2c", "cam_t_w2c" are saved in scene_camera.json. Default: True
          - bool
        * - ignore_dist_thres
          - Distance between camera and object after which object is ignored. Mostly due to failed physics. Default: 100.
          - float
        * - depth_scale
          - Multiply the uint16 output depth image with this factor to get depth in mm. Used to trade-off between depth accuracy
            and maximum depth value. Default corresponds to 65.54m maximum depth and 1mm accuracy. Default: 1.0
          - float
        * - m2mm
          - Original bop annotations and models are in mm. If true, we convert the gt annotations to mm here. This
            is needed if BopLoader option mm2m is used. Default: True
          - bool
    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)

        # Create CameraWriter to write camera attibutes, use OpenCV coordinates
        self._camera_writer = CameraStateWriter(config)
        self._camera_writer.destination_frame = ["X", "-Y", "-Z"]

        # Parse configuration.
        self.dataset = self.config.get_string("dataset", "")

        self.append_to_existing_output = self.config.get_bool("append_to_existing_output", True)

        # Save world to camera transformation
        self._save_world2cam = self.config.get_bool("save_world2cam", True)

        # Distance in meteres to object after which it is ignored. Mostly due to failed physics.
        self._ignore_dist_thres = self.config.get_float("ignore_dist_thres", 100.)

        # Multiply the output depth image with this factor to get depth in mm.
        self.depth_scale = self.config.get_float("depth_scale", 1.0)

        # Output translation gt in mm
        self._scale = 1000. if self.config.get_bool("m2mm", False) else 1.

        # Format of the depth images.
        depth_ext = '.png'

        # Debug mode flag
        self._avoid_rendering = config.get_bool("avoid_rendering", False)

        # Output paths.
        base_path = self._determine_output_dir(False)
        self.dataset_dir = os.path.join(base_path, 'data', self.dataset)
        self.camera_path = os.path.join(self.dataset_dir, 'camera.json')
        self.rgb_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'rgb', '{im_id:06d}' + '{im_type}')
        self.depth_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'depth', '{im_id:06d}' + depth_ext)
        self.chunk_camera_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'scene_camera.json')
        self.foreground_background_objects_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'foreground_background.json')
        self.segmentations_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'segmentation', '{im_id:06d}' + '.png')
        self.segmentations_labels_tpath = os.path.join(self.dataset_dir,'{chunk_id:06d}', 'segmentation_labels.csv')
        self.correspondances_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'correspondance', 'from_{im_id_from:06d}'+'_to_{im_id_to:06d}' + '.png')
        self.semantic_map_tpath = os.path.join(
            self.dataset_dir, '{chunk_id:06d}', 'correspondance_semantic_map',
            'from_{im_id_from:06d}' + '_to_{im_id_to:06d}' + '.png')

        # Create the output directory structure.
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        elif not self.append_to_existing_output:
            raise Exception("The output folder already exists: {}.".format(
                self.dataset_dir))



    def run(self):
        """ Stores frames and annotations for objects from the specified dataset.
        """

        all_mesh_objects = get_all_blender_mesh_objects()

        # Select objects from the specified dataset.

        self.dataset_objects = all_mesh_objects

        # Check if there is any object from the specified dataset.
        if not self.dataset_objects:
            raise Exception("The scene does not contain any object from the "
                            "specified dataset: {}. Either remove the dataset parameter "
                            "or assign custom property 'bop_dataset_name' to selected objects".format(self.dataset))

        # Get the camera.
        cam_ob = bpy.context.scene.camera
        self.cam = cam_ob.data
        self.cam_pose = (self.cam, cam_ob)

        # Save the data.
        self._write_camera()
        self._write_frames()

    def _write_camera(self):
        """ Writes camera.json into dataset_dir.
        """

        cam_K = self._camera_writer._get_attribute(self.cam_pose, 'cam_K')
        camera = {'cx': cam_K[0][2],
                  'cy': cam_K[1][2],
                  'depth_scale': self.depth_scale,
                  'fx': cam_K[0][0],
                  'fy': cam_K[1][1],
                  'height': bpy.context.scene.render.resolution_y,
                  'width': bpy.context.scene.render.resolution_x}

        save_json(self.camera_path, camera)

    def _get_frame_camera(self):
        """ Returns camera parameters for the active camera.

        :return: dict containing info for scene_camera.json
        """

        cam_K = self._camera_writer._get_attribute(self.cam_pose, 'cam_K')

        frame_camera_dict = {
            'cam_K': cam_K[0] + cam_K[1] + cam_K[2],
            'height': bpy.context.scene.render.resolution_y,
            'width': bpy.context.scene.render.resolution_x,
            'depth_scale': self.depth_scale
        }

        if self._save_world2cam:
            H_c2w_opencv = Matrix(self._camera_writer._get_attribute(self.cam_pose, 'cam2world_matrix'))

            H_w2c_opencv = H_c2w_opencv.inverted()
            R_w2c_opencv = H_w2c_opencv.to_quaternion().to_matrix()
            t_w2c_opencv = H_w2c_opencv.to_translation() * self._scale

            frame_camera_dict['cam_R_w2c'] = list(R_w2c_opencv[0]) + list(R_w2c_opencv[1]) + list(R_w2c_opencv[2])
            frame_camera_dict['cam_t_w2c'] = list(t_w2c_opencv)

        return frame_camera_dict

    def _write_frames(self):
        """ Writes images, GT annotations and camera info.
        """

        with Locker():
            # Paths to the already existing chunk folders (such folders may exist
            # when appending to an existing dataset).
            chunk_dirs = sorted(glob.glob(os.path.join(self.dataset_dir, '*')))
            chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d)]

            # Get ID's of the last already existing chunk and frame.
            curr_chunk_id = len(chunk_dirs)
            curr_frame_id = 0

            #Make subfolders to chunks
            os.makedirs(os.path.dirname(
                self.rgb_tpath.format(chunk_id=curr_chunk_id, im_id=0, im_type='PNG'))) # only takes the directory
            os.makedirs(os.path.dirname(
                self.depth_tpath.format(chunk_id=curr_chunk_id, im_id=0)))
            os.makedirs(os.path.dirname(
                self.segmentations_tpath.format(chunk_id=curr_chunk_id, im_id=0)))
            os.makedirs(os.path.dirname(
                self.correspondances_tpath.format(chunk_id=curr_chunk_id, im_id_from=0, im_id_to =0)))
            os.makedirs(os.path.dirname(
                self.semantic_map_tpath.format(chunk_id=curr_chunk_id, im_id_from=0, im_id_to=0)))

        # Initialize structures for the camera info.
        chunk_camera = {}
        chunk_depth_fpath = []



        # Go through all frames.
        num_new_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start
        end_frame = bpy.context.scene.frame_end if not self._avoid_rendering else bpy.context.scene.frame_start


        ####DEBUG ###
        # image = []
        ######

        for frame_id in range(bpy.context.scene.frame_start, end_frame):
            # Activate frame.
            bpy.context.scene.frame_set(frame_id)


            background_objects = []
            foreground_objects = []
            for obj in bpy.context.scene.objects:
                if("manip_object" in obj):
                    if(not obj['manip_object']):
                        background_objects.append(obj.name)

                    if(obj['manip_object']):
                        foreground_objects.append(obj.name)

            save_json(self.foreground_background_objects_tpath.format(chunk_id=curr_chunk_id), {"foreground_objects": foreground_objects,
                                    "background_objects": background_objects})


            ### DO RGB ###
            # Copy the resulting RGB image.
            rgb_output = Utility.find_registered_output_by_key("colors")
            if rgb_output is None:
                raise Exception("RGB image has not been rendered.")
            image_type = '.png' if rgb_output['path'].endswith('png') else '.jpg'
            rgb_fpath = self.rgb_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id, im_type=image_type)



            shutil.copyfile(rgb_output['path'] % frame_id, rgb_fpath)

            ####~DEBUG #####
            # im =cv2.imread(rgb_fpath,-1)
            # image.append(im)
            ########
            #####

            #### DO DEPTH ###
            # Load the resulting dist image.
            dist_output = Utility.find_registered_output_by_key("distance")
            if dist_output is None:
                raise Exception("Distance image has not been rendered.")
            depth, _, _ = self._load_and_postprocess(dist_output['path'] % frame_id, "distance")

            # Scale the depth to retain a higher precision (the depth is saved
            # as a 16-bit PNG image with range 0-65535).
            depth_mm = 1000.0 * depth  # [m] -> [mm]
            depth_mm_scaled = depth_mm / float(self.depth_scale)

            # Save the scaled depth image.
            depth_fpath = self.depth_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id)
            chunk_depth_fpath.append(depth_fpath)
            save_depth(depth_fpath, depth_mm_scaled)
            ########

            #### DO SEG MAP ####
            seg_output = Utility.find_registered_output_by_key("segmap")
            segmap = self._load_file(Utility.resolve_path(seg_output['path'] % frame_id))
            segmap_fpath = self.segmentations_tpath.format(chunk_id=curr_chunk_id, im_id=curr_frame_id)
            save_segmap(segmap_fpath,segmap)
            seg_labels = Utility.find_registered_output_by_key("segcolormap")
            seg_labels_fpath = self.segmentations_labels_tpath.format(chunk_id=curr_chunk_id)
            shutil.copyfile(seg_labels['path'], seg_labels_fpath)

            ##### DO Camera posemat ####
            # Get GT annotations and camera info for the current frame.
            chunk_camera[curr_frame_id] = self._get_frame_camera()
            # Save the chunk info if we are at the end of a chunk or at the last new frame.
            if (frame_id == num_new_frames - 1):
                # Save camera info.
                save_json(self.chunk_camera_tpath.format(chunk_id=curr_chunk_id), chunk_camera)

            else:
                curr_frame_id += 1

        #### DEBUG ####
        # average_pixel_value = np.mean([np.mean(np.asarray(i).astype(float)) for i in image])
        # average_pixel_std = np.std([np.mean(np.asarray(i).astype(float)) for i in image])
        # average_in_pixel_std = np.mean([np.std(np.asarray(i).astype(float)) for i in image])
        # print('AVERAGE_PIXEL VAL {}'.format(average_pixel_value))
        # print('AVERAGE_PIXEL std between images {}'.format(average_pixel_std))
        # print('AVERAGE_PIXEL std within images {}'.format(average_in_pixel_std))

        #########


        #### DO CORRESPONDANCES
        for i in range(len(chunk_depth_fpath)):
            for j in range(i+1, len(chunk_depth_fpath)):
            # j = i+1
            # if(j == len(chunk_depth_fpath)):
            #     continue
                self.get_and_save_correspondances(i,j,chunk_depth_fpath,chunk_camera,curr_chunk_id)

            # for j in range(i):
            #     self.get_and_save_correspondances(i, j, chunk_depth_fpath, chunk_camera, curr_chunk_id)



    def get_and_save_correspondances(self,i,j, chunk_depth_fpath, chunk_camera, curr_chunk_id ):
        current_d = cv2.imread(chunk_depth_fpath[i], -1).astype(np.float) / 1000.
        next_d = cv2.imread(chunk_depth_fpath[j], -1).astype(np.float) / 1000.
        current_cam_posemat = np.linalg.pinv(self.get_posemat_from_rot_transl(chunk_camera[i]['cam_R_w2c'],
                                                               chunk_camera[i]['cam_t_w2c']))
        next_cam_posemat =  np.linalg.pinv(self.get_posemat_from_rot_transl(chunk_camera[j]['cam_R_w2c'], chunk_camera[j]['cam_t_w2c']))
        cam_intrinsics = np.asarray(self._camera_writer._get_attribute(self.cam_pose, 'cam_K'))

        correspondances_i_to_j, semantics_map_i_to_j = self.get_correspondance_map(cam_intrinsics, current_cam_posemat,
                                                                                   next_cam_posemat, current_d, next_d)

        correspondances_fpath = self.correspondances_tpath.format(chunk_id=curr_chunk_id, im_id_from=i, im_id_to=j)
        semantics_fpath = self.semantic_map_tpath.format(chunk_id=curr_chunk_id, im_id_from=i, im_id_to=j)

        correspondances_i_to_j = np.concatenate(
            [correspondances_i_to_j, np.zeros((correspondances_i_to_j.shape[0], correspondances_i_to_j.shape[1], 1))],
            axis=2)
        correspondances_i_to_j[correspondances_i_to_j < 0.] = 0
        correspondances_i_to_j = correspondances_i_to_j.astype('uint16')

        semantics_map_i_to_j = semantics_map_i_to_j.astype("uint8")

        cv2.imwrite(correspondances_fpath, correspondances_i_to_j)
        cv2.imwrite(semantics_fpath, semantics_map_i_to_j)

    def get_posemat_from_rot_transl(self, rot, trans):
        posemat= np.zeros((4,4))
        posemat[3,3]=1.0
        posemat[:3,3]= trans
        posemat[:3,:3]= np.asarray(rot).reshape((3,3))

        return posemat

    def pixel_coord_np(self,height , width, ):
        """
        Pixel in homogenous coordinate
        Returns:
            Pixel coordinate:       [3, width * height]
        """
        cols = np.linspace(0, width - 1, width).astype(np.int)
        rows = np.linspace(0, height - 1, height).astype(np.int)
        return np.meshgrid(cols,rows)

    def get_correspondance_map(self, intrinsics, cam1_posemat, cam2_posemat, cam1_D, cam2_D):
        height = bpy.context.scene.render.resolution_y
        width = bpy.context.scene.render.resolution_x

        intrinsics_inv = np.linalg.pinv(intrinsics)
        cam1_extrinsics_inv = cam1_posemat

        cam2_extrinsics = T.pose_inv(cam2_posemat)

        # mask_x are the columns (so x in image coords) and mask_y are rows (so y in image coords)
        mask_cols, mask_rows = self.pixel_coord_np(cam1_D.shape[0],cam1_D.shape[1])
        mask_x = mask_cols.reshape((-1))
        mask_y = mask_rows.reshape((-1))

        # Get to world frame coordinates
        d_values = cam1_D[mask_y, mask_x]
        cam1_hom_image_coordinates = np.vstack([mask_x, mask_y, np.ones_like(mask_y)])
        cam1_frame_coords = intrinsics_inv[:3, :3].dot(cam1_hom_image_coordinates) * d_values.flatten()
        cam1_frame_coords = np.vstack([cam1_frame_coords, np.ones((1, cam1_frame_coords.shape[1]))])
        world_frame_coords = cam1_extrinsics_inv.dot(cam1_frame_coords)

        # Get to cam2 frame coordinates and then get correspondances to the cam2 image
        cam2_frame_coords = cam2_extrinsics.dot(world_frame_coords)
        cam2_intrinsics = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
        cam2_image_coords = cam2_intrinsics.dot(cam2_frame_coords)
        cam2_image_coords /= cam2_image_coords[-1, :]
        cam2_image_coords = np.round(cam2_image_coords).astype(int)[:2, :] # 0'th index is width (columns) and 1st index is height (rows)

        cam2_image_correspondances = cam2_image_coords.reshape((2,cam1_D.shape[0],cam1_D.shape[1])).transpose(1,2,0)

        #Select valid entries that project inside the image
        row0_selection = np.logical_and(cam2_image_coords[0, :] < width, cam2_image_coords[0, :] >= 0)
        row1_selection = np.logical_and(cam2_image_coords[1, :] < height, cam2_image_coords[1, :] >= 0)
        row_selection = np.logical_and(row0_selection, row1_selection)

        # Getting the out of frame map
        out_of_frame_map = np.logical_not(row_selection).reshape((cam1_D.shape[0],cam1_D.shape[1]))

        #Getting the occlusions map
        occlusions_mask_flattened = -1000* np.ones((cam1_D.shape[0]*cam1_D.shape[1])) #Making sure that there are no false positives by seeting a negative buffer array

        # For the in-frame correspondances, get the correspondance coordinates
        # cam2_image_coords_selected shows the x,y correspondance of each selected entry of a flattened cam1 image
        # To see which entry in a flattened cam 1 iamge is selected, we can use row_selection_where
        cam2_image_coords_selected = cam2_image_coords[:2,row_selection]
        row_selection_where = np.where(row_selection)

        # Get the depth values from cam2 by using the correspondance pixels
        # Put those depth values into the  occlusion mask buffer, note the buffer is in the cam1 frame so we use the row_selection_where
        vals_in_cam2_D = cam2_D[cam2_image_coords_selected[1,:],cam2_image_coords_selected[0,:]]
        occlusions_mask_flattened[row_selection_where] = vals_in_cam2_D

        # Reshape and compare what we get to the 3D coordinates of the points in the cam2 frame
        occlusions_mask = occlusions_mask_flattened.reshape(cam1_D.shape) #We need to get the depth values of cam2 for pixels in cam1 given the correspondances
        cam2_frame_coords_reshaped = cam2_frame_coords[2,:].reshape(cam1_D.shape) # The 3D coordinates in the cam2 frame for each pixel in cam1

        # Allow a tolerance of 1mm for coppelia inaccuracies
        occlusions_mask = np.isclose(occlusions_mask, cam2_frame_coords_reshaped, atol= 1.5e-3, rtol=0.).astype('uint8')

        #Combining the occlusions map and out of frame map
        semantics_map = 2*occlusions_mask +  out_of_frame_map*-1
        assert not np.isclose(semantics_map,1.).any() # sanity check
        semantics_map +=1

        # Semantics_map : 1 is occluded, 3  is visible, 0 is out of frame.


        #By hand it would be:
        #occlusions_mask =np.zeros(cam2_D.shape)
        # for i in range(cam2_D.shape[0]):
        #  for j in range(cam2_D.shape[1]):
        #  coords_in_cam2_D = cam2_image_correspondance[i,j]
        #  cam2_frame_coords_reshaped = cam2_frame_coords[2,:].reshape(cam2_D.shape)
        #  val_in_cam2_D = cam2_D[coords_in_cam2_D[1],coords_in_cam2_D[0]]
        #  val_in_world_frame = cam2_frame_coords_reshaped[i,j]
        #   occlusions_mask[i,j] = val_in_world_frame == val_in_cam2_D


        return cam2_image_correspondances, semantics_map