import json
import os
import math
import glob
import numpy as np
# import png
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
from src.utility.WriterUtility import WriterUtility
import matplotlib
import matplotlib.pyplot as plt
import pickle



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



class MyWriter:

    def __init__(self, base_path, dataset_dir_name, append_to_existing=True):
        self.append_to_existing = append_to_existing

        # Output paths.
        self.dataset_dir = os.path.join(base_path, 'data', dataset_dir_name)
        self.chunk_tdir =  os.path.join(self.dataset_dir, '{chunk_id:06d}')
        self.rgb_tpath = os.path.join(self.chunk_tdir, 'rgb', '{im_id:06d}' + '.png')
        self.depth_tpath = os.path.join(self.chunk_tdir, 'depth', '{im_id:06d}' + '.png')
        self.foreground_segmentation_tpath = os.path.join( self.chunk_tdir, 'foreground_segmentation', '{im_id:06d}' + '.png')
        self.segmentations_tpath = os.path.join(self.chunk_tdir, 'segmentation', '{im_id:06d}' + '.png')
        self.correspondances_tpath = os.path.join(self.chunk_tdir, 'correspondance', 'from_{im_id_from:06d}'+'_to_reference' + '.png')
        self.semantic_map_tpath = os.path.join(self.chunk_tdir, 'correspondance_semantic_map',   'from_{im_id_from:06d}' + '_to_reference' + '.png')

        self.segmentations_labels_tpath = os.path.join(self.chunk_tdir, 'segmentation_labels.pckl')
        self.chunk_camera_intrinsics_tpath = os.path.join(self.chunk_tdir, 'camera_intrinsics.npy')
        self.chunk_camera_pose_tpath = os.path.join(self.chunk_tdir, 'camera_poses.pckl')

        self.bottleneck_tpath = os.path.join(self.chunk_tdir,'bottleneck','{name}'+'{type}')


        # Create the output directory structure.
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        elif not self.append_to_existing:
            raise Exception("The output folder already exists: {}.".format(
                self.dataset_dir))


        self.chunk_id = None
        self.image_id = 0



    def _get_frame_camera(self):
        """ Returns camera parameters for the active camera.

        :return: dict containing info for scene_camera.json
        """

        cam_K = WriterUtility.get_cam_attribute(self.cam_pose[1], 'cam_K')

        frame_camera_dict = {
            'cam_K': cam_K[0] + cam_K[1] + cam_K[2],
            'height': bpy.context.scene.render.resolution_y,
            'width': bpy.context.scene.render.resolution_x,
            'depth_scale': self.depth_scale
        }

        if self._save_world2cam:
            H_c2w_opencv = Matrix(WriterUtility.get_cam_attribute(self.cam_pose[1], 'cam2world_matrix'))

            H_w2c_opencv = H_c2w_opencv.inverted()
            R_w2c_opencv = H_w2c_opencv.to_quaternion().to_matrix()
            t_w2c_opencv = H_w2c_opencv.to_translation() * self._scale

            frame_camera_dict['cam_R_w2c'] = list(R_w2c_opencv[0]) + list(R_w2c_opencv[1]) + list(R_w2c_opencv[2])
            frame_camera_dict['cam_t_w2c'] = list(t_w2c_opencv)

        return frame_camera_dict


    def check_and_create_trajectory_folders(self):

        with Locker():
            # Paths to the already existing chunk folders (such folders may exist
            # when appending to an existing dataset).
            chunk_dirs = sorted(glob.glob(os.path.join(self.dataset_dir, '*')))
            chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d)]

            # Get ID's of the last already existing chunk and frame.
            curr_chunk_id = len(chunk_dirs)

            chunk_dir = self.chunk_tdir.format(chunk_id=curr_chunk_id)
            if(not os.path.exists(chunk_dir)):
                os.makedirs(os.path.dirname(
                    self.rgb_tpath.format(chunk_id=curr_chunk_id, im_id=0, im_type='.png')))  # only takes the directory
                os.makedirs(os.path.dirname(
                    self.depth_tpath.format(chunk_id=curr_chunk_id, im_id=0)))
                os.makedirs(os.path.dirname(
                    self.segmentations_tpath.format(chunk_id=curr_chunk_id, im_id=0)))
                # os.makedirs(os.path.dirname(
                #     self.segmentations_labels_tpath.format(chunk_id=curr_chunk_id, im_id=0)))
                os.makedirs(os.path.dirname(
                    self.correspondances_tpath.format(chunk_id=curr_chunk_id, im_id_from=0)))
                os.makedirs(os.path.dirname(
                    self.semantic_map_tpath.format(chunk_id=curr_chunk_id, im_id_from=0)))
                os.makedirs(os.path.dirname(
                    self.foreground_segmentation_tpath.format(chunk_id=curr_chunk_id, im_id=0)))

                os.makedirs(os.path.dirname(
                    self.bottleneck_tpath.format(chunk_id=curr_chunk_id, name='rgb', type ='.png')))

            self.chunk_id =curr_chunk_id
        return curr_chunk_id


    def update_image_id(self):
        self.image_id+=1

    def save_rgb(self, rgb_image, frame_id = None, chunk_id = None, is_bottleneck = False):
        if(frame_id is None):
            frame_id = self.image_id

        if(chunk_id is None):
            chunk_id = self.chunk_id

        if(is_bottleneck):
            rgb_fpath = self.bottleneck_tpath.format(chunk_id=chunk_id, name='rgb', type='.png')
        else:
            rgb_fpath = self.rgb_tpath.format(chunk_id=chunk_id, im_id=frame_id)

        cv2.imwrite(rgb_fpath, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    def save_depth(self, depth_image,  frame_id = None, chunk_id = None, is_bottleneck = False):
        if (frame_id is None):
            frame_id = self.image_id

        if (chunk_id is None):
            chunk_id = self.chunk_id

        depth_mm = np.round(1000.0 * depth_image).astype('uint16')
        if(is_bottleneck):
            depth_fpath = self.bottleneck_tpath.format(chunk_id=chunk_id, name='depth', type='.png')
        else:
            depth_fpath = self.depth_tpath.format(chunk_id=chunk_id, im_id=frame_id)

        cv2.imwrite(depth_fpath, depth_mm)
        # plt.imsave(depth_fpath, depth_mm)


    def save_seg_map(self, seg_data, frame_id = None, chunk_id = None,is_bottleneck = False):
        if (frame_id is None):
            frame_id = self.image_id

        if (chunk_id is None):
            chunk_id = self.chunk_id

        # TODO: extend to many seg maps and many attributes
        seg_map = seg_data['instance_segmaps'][0]
        attribute = seg_data['instance_attribute_maps'][0]

        if(is_bottleneck):
            segmap_fpath = self.bottleneck_tpath.format(chunk_id=chunk_id, name='seg_map', type='.png')
        else:
            segmap_fpath = self.segmentations_tpath.format(chunk_id=chunk_id, im_id=frame_id)

        cv2.imwrite(segmap_fpath, seg_map)
        # plt.imsave(segmap_fpath, seg_map)

        seg_labels_fpath = self.segmentations_labels_tpath.format(chunk_id=chunk_id, im_id=frame_id)

        if (os.path.exists(seg_labels_fpath)):
            seg_labels = pickle.load((open(seg_labels_fpath, 'rb')))
        else:
            seg_labels = {}


        if(is_bottleneck):
            seg_labels["bottleneck"] = attribute

        else:
            seg_labels["{:06d}".format(frame_id)] = attribute

        pickle.dump(seg_labels, open(seg_labels_fpath,'wb'))

    def save_foreground_seg_map(self, seg_map, frame_id = None, chunk_id = None, is_bottleneck = False):
        if (frame_id is None):
            frame_id = self.image_id

        if (chunk_id is None):
            chunk_id = self.chunk_id

        if(is_bottleneck):
            segmap_fpath = self.bottleneck_tpath.format(chunk_id=chunk_id, name='foreground_seg_map', type='.png')
        else:
            segmap_fpath = self.foreground_segmentation_tpath.format(chunk_id=chunk_id, im_id=frame_id)

        cv2.imwrite(segmap_fpath, seg_map)
        # plt.imsave(segmap_fpath, seg_map)

    def save_camera_intrinsics(self, intrinsics_matrix, chunk_id = None):
        if (chunk_id is None):
            chunk_id = self.chunk_id

        intrinsics_fpath = self.chunk_camera_intrinsics_tpath.format(chunk_id=chunk_id)
        np.save(intrinsics_fpath, intrinsics_matrix)


    def save_camera_pose(self, pose, frame_id = None, chunk_id = None, is_bottleneck = False):
        if (frame_id is None):
            frame_id = self.image_id

        if (chunk_id is None):
            chunk_id = self.chunk_id

        camera_pose_fpath = self.chunk_camera_pose_tpath.format(chunk_id=chunk_id)

        if(os.path.exists(camera_pose_fpath)):
            camera_poses = pickle.load((open(camera_pose_fpath,'rb')))
        else:
            camera_poses = {}

        if(is_bottleneck):
            camera_poses["bottleneck"]=pose

        else:
            camera_poses["{:06d}".format(frame_id)]=pose

        pickle.dump(camera_poses,open(camera_pose_fpath,'wb'))



    def get_and_save_correspondances(self,i,j, chunk_depth_fpath, chunk_camera, curr_chunk_id ):
        current_d = cv2.imread(chunk_depth_fpath[i], -1).astype(np.float) / 1000.
        next_d = cv2.imread(chunk_depth_fpath[j], -1).astype(np.float) / 1000.
        current_cam_posemat = np.linalg.pinv(self.get_posemat_from_rot_transl(chunk_camera[i]['cam_R_w2c'],
                                                               chunk_camera[i]['cam_t_w2c']))
        next_cam_posemat =  np.linalg.pinv(self.get_posemat_from_rot_transl(chunk_camera[j]['cam_R_w2c'], chunk_camera[j]['cam_t_w2c']))
        cam_intrinsics = np.asarray(WriterUtility.get_cam_attribute(self.cam_pose[1], 'cam_K'))

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