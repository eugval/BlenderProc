import json
import os
import glob
import numpy as np
from src.utility.LockerUtility import Locker
from src.Eugene.dataset_utils import cam_from_blender_to_opencv

import bpy
from src.utility import TransformUtility as T
import cv2
import pickle
import matplotlib.pyplot as plt



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



class MyNewWriter:

    def __init__(self, base_path, dataset_dir_name, append_to_existing=True, has_bottleneck = False):
        self.append_to_existing = append_to_existing

        # Output paths.
        self.dataset_dir = os.path.join(base_path,  dataset_dir_name)
        self.chunk_tdir =  os.path.join(self.dataset_dir, '{chunk_id:06d}')

        # Create the output directory structure.
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        elif not self.append_to_existing:
            raise Exception("The output folder already exists: {}.".format(
                self.dataset_dir))


        self.chunk_id = None
        self.image_id = 0


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
                os.makedirs(chunk_dir)

            self.chunk_id = curr_chunk_id
        return curr_chunk_id


    def update_image_id(self):
        self.image_id+=1

    def make_path(self, folder, file_idx = None, extension = None,  chunk_id = None, file_name = None):
        if (chunk_id is None):
            chunk_id = self.chunk_id

        if(extension is None):
            extension = '.png'

        if(file_idx is None):
            file_idx = self.image_id

        if(file_name is not None):
            path =  os.path.join(self.chunk_tdir.format(chunk_id = chunk_id), '{}'.format(folder), file_name)
        else:
            path =  os.path.join(self.chunk_tdir.format(chunk_id = chunk_id), '{}'.format(folder), '{:06d}'.format(file_idx) + '{}'.format(extension))

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        return path

    def save_rgb(self, rgb_image, path = None):
        cv2.imwrite(path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        # plt.imsave(path,rgb_image)

    def save_normals(self, normals_image, path=None):
        normals_int = np.round(1000.0 * normals_image).astype('uint16')
        cv2.imwrite(path, normals_int)
        # plt.imsave(path,normals_image)

    def save_depth(self, depth_image,  path = None):
        depth_mm = np.round(1000.0 * depth_image).astype('uint16')
        cv2.imwrite(path, depth_mm)
        # plt.imsave(path, depth_image)

    def save_distance(self, dist_image,  path = None):
        dist_mm = np.round(1000.0 * dist_image).astype('uint16')
        cv2.imwrite(path, dist_mm)
        # plt.imsave(path, dist_mm)

    def save_seg_map(self, seg_map, seg_metadata, segment_by, values,   path, resize_size = None, check_degenerate = True):


        if(isinstance(values,str)):
            values = [values]


        ids_to_segment = []

        for bundle in  seg_metadata:
            if(bundle[segment_by] in values):
                ids_to_segment.append(bundle['idx'])

        final_seg_map = np.zeros_like(seg_map)

        for id_to_segment in ids_to_segment:
            final_seg_map += (seg_map == id_to_segment).astype('uint8')


        final_seg_map  = (final_seg_map>0).astype('uint8')

        if(resize_size is not None):
            final_seg_map = cv2.resize(final_seg_map, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)


        if(check_degenerate):
            self.check_degenerate(final_seg_map)

        cv2.imwrite(path, final_seg_map)
        # plt.imsave(path, final_seg_map)

    def save_full_seg_map(self, seg_map, seg_metadata,  path, resize_size = None):
        ids_of_background = []

        for bundle in  seg_metadata:
            if(bundle['category_id'] in 'background' or bundle['name'] in ['ground_plane0', 'ground_plane1', 'ground_plane2', 'ground_plane3', 'ground_plane4', 'ground_plane5']):
                ids_of_background.append(bundle['idx'])


        for bg_id in ids_of_background:
            seg_map[seg_map == bg_id]=0.


        seg_map  = seg_map.astype('uint8')

        if(resize_size is not None):
            seg_map = cv2.resize(seg_map, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(path, seg_map)

    def check_degenerate(self, seg ):
        pos = np.where(seg)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        box = np.array([xmin, ymin, xmax, ymax])

        degenerate_box = box[2:] <= box[:2]
        if (degenerate_box.any()):
            raise ValueError('IS DEGENERATE seg!')

    def save_camera_pose(self, pose, pose_path, pose_key):
        if(os.path.exists(pose_path)):
            camera_poses = pickle.load((open(pose_path,'rb')))
        else:
            camera_poses = {}

        camera_poses[pose_key] = pose

        pickle.dump(camera_poses,open(pose_path,'wb'))


    def save_camera_intrinsics(self, intrinsics_matrix, path):
        np.save(path, intrinsics_matrix)

    def save_object_info(self, object_info, path):
        pickle.dump(object_info, open(path,'wb'))




    def get_and_save_correspondances(self,i,j, depths, camera_poses , cam_intrinsics, correspondence_path, semantic_map_path, depth_for_j=None, camera_poses_for_j=None):
        current_d = depths[i]
        current_cam_posemat = camera_poses[i]
        if(depth_for_j is not None):
            next_d = depth_for_j[j]
            next_cam_posemat = camera_poses_for_j[j]
        else:
            next_d = depths[j]
            next_cam_posemat = camera_poses[j]
        #
        correspondances_i_to_j, semantics_map_i_to_j = self.get_correspondance_map(cam_intrinsics, current_cam_posemat,
                                                                                   next_cam_posemat, current_d, next_d)



        correspondances_i_to_j = np.concatenate(
            [correspondances_i_to_j, np.zeros((correspondances_i_to_j.shape[0], correspondances_i_to_j.shape[1], 1))],
            axis=2)
        correspondances_i_to_j[correspondances_i_to_j < 0.] = 0
        correspondances_i_to_j = correspondances_i_to_j.astype('uint16')
        semantics_map_i_to_j = semantics_map_i_to_j.astype("uint8")

        cv2.imwrite(correspondence_path, correspondances_i_to_j)
        cv2.imwrite(semantic_map_path, semantics_map_i_to_j)
        return correspondances_i_to_j, semantics_map_i_to_j


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
        height = cam1_D.shape[0]
        width = cam1_D.shape[1]

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