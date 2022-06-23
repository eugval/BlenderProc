from src.utility.SetupUtility import SetupUtility
argv = SetupUtility.setup(['matplotlib','opencv-contrib-python', 'scikit-learn', 'numpy==1.20'  ,'numba', 'scipy', ]) #'pydevd-pycharm~=193.6911.2'

import sys
print(sys.argv)

from src.utility.Utility import Utility
from src.utility.CameraUtility import CameraUtility

from src.utility.WriterUtility import WriterUtility
from src.utility.Initializer import Initializer
from src.utility.loader.BlendLoader import BlendLoader
from src.utility.LightUtility import Light
from src.utility.MathUtility import MathUtility
from src.utility.MeshObjectUtility import MeshObject
from src.utility.lighting.SurfaceLighting import SurfaceLighting
from src.utility.RendererUtility import RendererUtility
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from examples.visual_servo.my_writer import MyWriter
from examples.visual_servo import se3_tools as se3

import numpy as np
import os
import bpy
import colorsys
import matplotlib.pyplot as plt
from dense_correspondence_control.control.visual_servo import get_visual_servoing,do_registration_with_correspondances_ransac
from dense_correspondence_control.control.find_correspondences import put_optical_flow_arrows_on_image


temp_save_folder = '/home/eugene/test/'


def run_modules(modules):
    for module in modules:
        module.run()


def aggregate_segmap_values(key,value, seg_data):
    #TODO: extend to many seg maps and many attributes
    seg_map = seg_data['instance_segmaps'][0]
    attribute = seg_data['instance_attribute_maps'][0]

    indices_to_merge =set()

    for object in attribute:
        if (object[key]==value):
            indices_to_merge.add(object['idx'])

    seg_map_new = np.zeros_like(seg_map)
    for i in indices_to_merge:
        seg_map_new+=seg_map==i



    return seg_map_new


def cam_from_blender_to_opencv(blender_cam_posemat):
    # R_opencv_to_blender = np.array([[1, 0, 0],
    #                                 [0, -1, 0],
    #                                 [0, 0, -1]]) #### I THINK THIS MATRIX IS WRONG
    #
    # rot_blender = blender_cam_posemat[:3,:3]
    # rot_opencv = R_opencv_to_blender @ rot_blender
    #
    # new_pose = np.eye(4)
    # new_pose[:3,:3]=rot_opencv
    # new_pose[:3,3]= blender_cam_posemat[:3,3]
    new_pose = MathUtility.change_source_coordinate_frame_of_transformation_matrix(blender_cam_posemat,  ["X", "-Y", "-Z"])
    return new_pose


def spherical_to_cartesian( phi, theta, r=1.0):
    ''' Returns the cartesian coordinates of spherical coordinates phi,theta,r
    phi : x-y plane angle in [-pi,pi]
    theta: angle from z-anxis in [0, pi]
    r: radius in [0, inf]
    '''
    return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

def sample_initial_camera_pose(centre_in_world,radius_range,  elevation_range, azimuth_range, look_at_rotation_min, look_at_rotation_max):
    r = np.random.uniform(*radius_range) # from 0 to inf
    e = np.random.uniform(*elevation_range) # theta , from 0 to pi
    a = np.random.uniform(*azimuth_range) # phi from 0 to 2pi

    position = centre_in_world +spherical_to_cartesian(a,e,r)
    residual_rot  = np.random.uniform(look_at_rotation_min,look_at_rotation_max)


    orientation = CameraUtility.rotation_from_forward_vec( centre_in_world-position,up_axis ='Y',
                                                           residual_rot=residual_rot)

    pose = np.eye(4)
    pose[:3,3]=position
    pose[:3,:3]=orientation

    return pose

def  apply_twist(current_pose, twist, timestep = 0.1):
    '''
    Applies the twist. twist is in the world frame so it is applied before the current pose.

    current_pose = Twc  Twist = T_delta in world
    so new pose = exp(Twis*time).dot(Twc)
    note the if the twist was in the ee frame then we would have post-multiplication.
    http://publish.illinois.edu/ece470-intro-robotics/files/2019/09/09-lecture.pdf
    '''
    delta_pose = twist * timestep

    delta_pose = se3.se3_exp(delta_pose)

    return current_pose.dot(delta_pose)
    #
    # return delta_pose.dot(current_pose)




def sample_color(h_range, s_range, v_range, rgba = False):
    h = np.random.uniform(*h_range)
    s = np.random.uniform(*s_range)
    v = np.random.uniform(*v_range)

    rgb = colorsys.hsv_to_rgb(h,s,v)
    if(rgba):
        return list(rgb)+[1.0]
    else:
        return list(rgb)


def loguniform_sampling(min,max, size = None):
    return np.asarray(np.exp(np.random.uniform(np.log(min), np.log(max), size)))



def fix_uv_maps():
    uv_fix_loading_module = Utility.initialize_modules([
        {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "type": "MESH"
                    }
                },
                "cf_add_uv_mapping": {
                    "projection": "cube",
                    "forced_recalc_of_uv_maps": True
                }
            }
        }
    ])
    run_modules(uv_fix_loading_module)


def randomise_materials():
    material_randomiser_loading_module = Utility.initialize_modules([{
        "module": "manipulators.MaterialRandomiser",
        "config": {
            "selector": {
                "provider": "getter.Material",
                "conditions": {
                    "cf_use_materials_of_objects": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "type": "MESH",
                        }
                    }
                }
            },
            "number_of_samples": 1,
            "parameters_to_randomise": ["base_color", "roughness", "metallic", "specular", "anisotropic", "sheen",
                                        "clearcoat"],
            "randomisation_probabilities": [1.0, 0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
            "metallic_min": 0.5,
            "anisotropic_min": 0.5,
            "clearcoat_min": 0.5,
            "clearcoat_roughness_min": 0.5,
            "sheen_min": 0.5,
            "keep_base_color": False,
            "displacement_probability": 0.5,
            "randomisation_types": ["monochrome_random"],
            "randomisation_type_probabilities": [1.0],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },])

    run_modules(material_randomiser_loading_module)

def randomise_visuals():
    # randomise ambient light - just
    emission_strength_sample = loguniform_sampling(0.1, 20)
    emission_color = [1.0,1.0,1.0,1.0]
    ambient_light_plane = MeshObject(bpy.data.objects['light_plane'])
    SurfaceLighting.run([ambient_light_plane],emission_strength = emission_strength_sample, emission_color=emission_color)

    # randomise directional light
    # sample location
    location1 = np.random.uniform([-3,-3,0],[3,3,3])
    # sample colour
    colour = sample_color([0.,1.0],[0.2,1.0],[0.4,1.0])
    # sample energy
    energy1 = np.random.uniform(1,350)

    light1 = Light()
    light1.set_type("POINT")
    light1.set_location(location1)
    light1.set_color(colour)
    light1.set_energy(energy1)

    # sample location
    location2 = np.random.uniform([-3,-3,0],[3,3,3])
    # sample colour

    # sample energy
    energy2 = np.random.uniform(1, 350)

    light2 = Light()
    light2.set_type("POINT")
    light2.set_location(location2)
    light2.set_color(colour)
    light2.set_energy(energy2)

    # randomise materials
    randomise_materials()


def gt_new_camera_pose(target_pose, current_pose, remaining_steps =1):
    delta_pose = np.matmul( target_pose, se3.pose_inv(current_pose.astype(float)))
    delta_pose_log = se3.se3_log(delta_pose)
    delta_pose_log = delta_pose_log/remaining_steps

    delta_pose =  se3.se3_exp(delta_pose_log)

    new_camera_pose = np.matmul(delta_pose, current_pose)
    return new_camera_pose


def render_scene(num_frame, chunk_id, camera_pose, is_bottleneck=False):
    # render
    data = RendererUtility.render(load_keys={'colors', 'depth'})

    # Run Rendering Segmentation
    seg_data = SegMapRendererUtility.render(map_by=["instance", "class"])

    seg_map_cups = aggregate_segmap_values('category_id', 'cup', seg_data)

    # Save RGB
    writer.save_rgb(data['colors'][0], frame_id=num_frame, chunk_id=chunk_id, is_bottleneck=is_bottleneck)
    writer.save_depth(data['depth'][0], frame_id=num_frame, chunk_id=chunk_id, is_bottleneck=is_bottleneck)
    writer.save_foreground_seg_map(seg_map_cups, frame_id=num_frame, chunk_id=chunk_id, is_bottleneck=is_bottleneck)
    writer.save_seg_map(seg_data, frame_id=num_frame, chunk_id=chunk_id, is_bottleneck=is_bottleneck)
    writer.save_camera_pose(camera_pose, frame_id=num_frame, chunk_id=chunk_id, is_bottleneck=is_bottleneck)

    return data, seg_data, seg_map_cups




def process_bottleneck(num_frame, chunk_id, pos_volume = None, orn_volume = None):
    # Place camera at bottleneck
    # Get object of interest
    manip_object = MeshObject('cup_0013')
    # Get the top point of its bounding box
    top_point = manip_object.get_top_mean_point()

    # Sample the bottleneck pose
    if(pos_volume is None):
        pos_delta = np.random.uniform([-0.02, -0.02, 0.11], [0.02, 0.02, 0.13])
    else:
        pos_delta = np.random.uniform(*pos_volume)

    bottleneck_pos = top_point + pos_delta

    if(orn_volume is None):
        residual_rot = np.random.uniform([-0.2618 / 2, -0.2618 / 2, -0.7854], [0.2618 / 2, 0.2618 / 2, 0.7854])
    else:
        residual_rot =  np.random.uniform(*orn_volume)

    orientation = CameraUtility.rotation_from_forward_vec(top_point - bottleneck_pos, up_axis='Y',
                                                          residual_rot=residual_rot)

    bottleneck_pose = np.eye(4)
    bottleneck_pose[:3, 3] = bottleneck_pos
    bottleneck_pose[:3, :3] = orientation
    CameraUtility.add_camera_pose(bottleneck_pose, 0)

    # render
    data, seg_data, seg_map_cups = render_scene(num_frame, chunk_id,bottleneck_pose, is_bottleneck =True)

    # return bottleneck stuff
    return bottleneck_pose, data['colors'][0], data['depth'][0], seg_map_cups




def pixel_coord_np(height , width, ):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    cols = np.linspace(0, width - 1, width).astype(np.int)
    rows = np.linspace(0, height - 1, height).astype(np.int)
    return np.meshgrid(cols,rows)


def get_correspondance_map(intrinsics, cam1_posemat, cam2_posemat, cam1_D, cam2_D):
    height,width = cam1_D.shape

    intrinsics_inv = np.linalg.pinv(intrinsics)
    cam1_extrinsics_inv = cam1_posemat# se3.pose_inv(cam1_posemat ) #

    cam2_extrinsics =se3.pose_inv(cam2_posemat) # cam2_posemat #

    # mask_x are the columns (so x in image coords) and mask_y are rows (so y in image coords)
    mask_cols, mask_rows = pixel_coord_np(height,width)
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

    cam2_image_correspondances = cam2_image_coords.reshape((2,cam1_D.shape[0],cam1_D.shape[1])).transpose(1,2,0) # 0'th channel is width (x, columns)

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
    occlusions_mask = np.isclose(occlusions_mask, cam2_frame_coords_reshaped, atol= 1.e-3, rtol=0.).astype('uint8')

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





Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/visual_servo" + "/"


Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('scene/room.blend')

# LOAD THE OBJECT INTO THE SCENE
object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ModelNetLoader",
        "config": {
          "data_path": '../../resources/ModelNet40',
          "model_name": "cup_0013",
           # "scale_path": "./per_category_scale.json",
            "manual_scale": 0.06,
          "train": True,
          "samples" : 1,
          "replacement": False,
          "add_properties": {
            "cp_physics": True,
            "cp_manip_object":True,
          },
          "cf_set_shading": "SMOOTH"
        }
      },

])

run_modules(object_loading_module)




# Define Visuals
fix_uv_maps()
randomise_visuals()



# define the camera intrinsics
intrinsics= np.array([[383.524322509766, 0., 318.551879882812],
                     [0, 383.524322509766, 238.973526000977],
                     [0., 0., 1.]])

CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 640, image_height=480,clip_start=0.001)
# CameraUtility.set_intrinsics_from_blender_params(1, 640/2, 480/2, lens_unit="FOV", clip_start= 0.0001)
# intrinsics = CameraUtility.get_intrinsics_as_K_matrix()




#Initialise Writer
writer = MyWriter(Utility.working_dir , 'test_trajectory')
chunk_id = writer.check_and_create_trajectory_folders()
writer.save_camera_intrinsics(CameraUtility.get_intrinsics_as_K_matrix(), chunk_id=chunk_id)

# activate depth rendering
RendererUtility.set_samples(350)
RendererUtility.enable_depth_output()


bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1



pos_volume =  ([0.,0.,0.15],[0.,0.,0.15]) #  ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0.,0.,0.],[0.,0.,0.]) # ([0.,0.,0.],[0.,0.,0.])  ([-0.08726, -0.08726, -0.7854], [0.08726, 0.08726, 0.7854]) #
bottleneck_pose, bottleneck_rgb, bottleneck_depth, bottleneck_seg_map = process_bottleneck(num_frame=0, chunk_id=chunk_id, pos_volume = pos_volume, orn_volume= orn_volume)

print('bottleneck pose!!! {}'.format(bottleneck_pose))

centre_in_world = np.array([0.,0.,0.])
elevation_range = [0.,   0.3490658503988659]  # (20 degrees)  0.3490658503988659
azimuth_range = [0.,2*np.pi]
radius_range = [0.2,0.3]
extra_rotation_min =[-0.17453, -0.17453,-0.7854] # [0, 0,-0.7854]
extra_rotation_max = [0.17453, 0.17453, 0.7854] # [0,0, 0.7854]

# #
# camera_pose = sample_initial_camera_pose(centre_in_world=centre_in_world,
#                                                  radius_range=radius_range,
#                                                  elevation_range=elevation_range,
#                                                  azimuth_range=azimuth_range,
#                                                  look_at_rotation_min=extra_rotation_min,
#                                                  look_at_rotation_max= extra_rotation_max)

cam_pos_delta = np.array([0.06,-0.02, 0.3]) #np.array([0.01,-0.01, 0.005])
cam_rot_delta = se3.so3_exp(np.array([0.,0., 0.9]).astype(float))  #array([0.01,-0.02,0.01]).astype(float))
cam_pose_delta = se3.make_pose(cam_pos_delta,cam_rot_delta)
camera_pose = np.matmul(cam_pose_delta,bottleneck_pose)


print('Initial_pose {}'.format(camera_pose))

#Initial state
CameraUtility.add_camera_pose(camera_pose, 0)
data, seg_data, seg_map_cups = render_scene(0, chunk_id, camera_pose, is_bottleneck=False)

cam2_image_correspondances, semantics_map = get_correspondance_map(CameraUtility.get_intrinsics_as_K_matrix(),
                                                                       cam_from_blender_to_opencv(camera_pose),
                                                                       cam_from_blender_to_opencv(bottleneck_pose),
                                                                       data['depth'][0], bottleneck_depth)
num_steps = 400
camera_pos_errors =[]
camera_pos_errors_x =[]
camera_pos_errors_y =[]
camera_pos_errors_z =[]
average_flows_x = []
average_flows_y = []
average_flows = []
rot_errors=[]
v_errors = []
for num_frame in range(1, num_steps):
    # matrix_world = MathUtility.build_transformation_mat(camera_pos, camera_euler)

    delt = bottleneck_pose[:3,3]-camera_pose[:3,3]
    camera_pos_error = np.linalg.norm(delt)
    camera_pos_error_x = np.abs(delt[0])
    camera_pos_error_y = np.abs(delt[1])
    camera_pos_error_z = np.abs(delt[2])
    camera_pos_errors.append(camera_pos_error)
    camera_pos_errors_y.append(camera_pos_error_y)
    camera_pos_errors_x.append(camera_pos_error_x)
    camera_pos_errors_z.append(camera_pos_error_z)
    plt.plot(camera_pos_errors, label = 'norm')
    plt.plot(camera_pos_errors_x, label = 'x')
    plt.plot(camera_pos_errors_y, label = 'y')
    plt.plot(camera_pos_errors_z, label = 'z')
    plt.legend()
    plt.ylim((np.min(camera_pos_errors+camera_pos_errors_x+camera_pos_errors_y+camera_pos_errors_z), np.max(camera_pos_errors+camera_pos_errors_x+camera_pos_errors_y+camera_pos_errors_z)))
    plt.savefig(temp_save_folder+'plot_position_error.png')
    plt.close()

    camera_rot_delta = bottleneck_pose[:3,:3]@camera_pose[:3,:3].T
    camera_rotvec_delta = se3.rot2rotvec(camera_rot_delta)
    rot_error = np.linalg.norm(camera_rotvec_delta)
    rot_errors.append(rot_error)
    plt.plot(rot_errors)
    plt.ylim((np.min(rot_errors), np.max(rot_errors)))
    plt.savefig(temp_save_folder + 'plot_rotation_error.png')
    plt.close()


    visible_object_pixels = seg_map_cups * (semantics_map==3).astype('uint8')
    flattened_correspondences = cam2_image_correspondances[visible_object_pixels>0.]
    flattened_depth = data['depth'][0][visible_object_pixels>0.]
    flattened_bottleneck_depth = bottleneck_depth[visible_object_pixels>0.]
    visible_object_pixels_selected = np.where(visible_object_pixels > 0.)
    starting_pixels = np.stack([visible_object_pixels_selected[1], visible_object_pixels_selected[0]],axis=1)

    f = flattened_correspondences - starting_pixels
    average_flows_y.append(np.mean(f[:,0]))
    average_flows_x.append(np.mean(f[:, 1]))
    average_flows.append(np.mean(np.linalg.norm(f,axis=1)))
    plt.plot(average_flows_y, label ='y')
    plt.plot(average_flows_x, label = 'x')
    plt.plot(average_flows, label = 'xy_norm')
    plt.ylim((np.min(average_flows_y+average_flows_x+average_flows), np.max(average_flows_y+average_flows_x+average_flows)))
    plt.legend()
    plt.savefig(temp_save_folder+'plot_flow.png')
    plt.close()



    image_test = put_optical_flow_arrows_on_image(data['colors'][0], cam2_image_correspondances, mask = visible_object_pixels, threshold=1.0)
    plt.imsave(writer.chunk_tdir.format(chunk_id=chunk_id)+'/{}.png'.format(num_frame), image_test)

    T_bottleneck_in_initial = do_registration_with_correspondances_ransac(data['depth'][0], bottleneck_depth,
                                                                          intrinsics, cam2_image_correspondances,
                                                                          segmentation_map=seg_map_cups)

    twist = se3.se3_log(T_bottleneck_in_initial) / 10


    camera_pose_initial_opencv = cam_from_blender_to_opencv(camera_pose)
    step = 1.0 # -  (num_frame/num_steps)*0.195
    camera_pose_final_opencv = apply_twist(camera_pose_initial_opencv,twist, step)
    camera_pose = cam_from_blender_to_opencv(camera_pose_final_opencv)

    # canera_pose = apply_twist(camera_pose, twist, 0.1)

    # camera_pose = gt_new_camera_pose(bottleneck_pose, camera_pose, remaining_steps=num_steps - num_frame)

    CameraUtility.add_camera_pose(camera_pose, num_frame)
    camera_pose_in_world = np.array(WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix'))

    bpy.context.scene.frame_start = num_frame
    bpy.context.scene.frame_end = num_frame + 1

    # Run Rendering Depth, Normals, RGB
    data, seg_data, seg_map_cups = render_scene(num_frame, chunk_id, camera_pose, is_bottleneck =False)
    cam2_image_correspondances, semantics_map = get_correspondance_map(CameraUtility.get_intrinsics_as_K_matrix(),
                                                                       cam_from_blender_to_opencv(camera_pose),
                                                                       cam_from_blender_to_opencv(bottleneck_pose),
                                                                       data['depth'][0], bottleneck_depth)


    # Calculate correspondenses and other stuff

    # Calculate new camera Image
    # Grab current camera pose
    # Grab reference camera pose

    # Calculate the ground truth difference between the two
    # Save the gt difference between the two in a file

    # Calculate the new camera pose using correspondances





    # camera_pose = sample_initial_camera_pose(centre_in_world=centre_in_world,
    #                                          radius_range=radius_range,
    #                                          elevation_range=elevation_range,
    #                                          azimuth_range=azimuth_range,
    #                                          look_at_rotation_min=extra_rotation_min,
    #                                          look_at_rotation_max=extra_rotation_max)













########## DEBUGGING CORRESPONDANCES

##############################################################
#
# camera_pose = sample_initial_camera_pose(centre_in_world=centre_in_world,
#                                                  radius_range=radius_range,
#                                                  elevation_range=[0.,0.],
#                                                  azimuth_range=[0.,0.,],
#                                                  look_at_rotation_min=[0.,0.,0.],
#                                                  look_at_rotation_max= [0.,0.,0.])
#
# intrinsics = CameraUtility.get_intrinsics_as_K_matrix()
#
# cam1_posemat = camera_pose
# cam_pos_delta = np.array([0.01,-0.02, -0.02])
# cam_rot_delta = se3.so3_exp(np.array([0.1,-0.2,0.6]).astype(float))
# cam_pose_delta = se3.make_pose(cam_pos_delta,cam_rot_delta)
# cam2_posemat = np.matmul(cam_pose_delta,cam1_posemat)
# x_in_world = np.array([0.,0.,0.])
# x_in_world_hom = np.array([0.,0.,0.,1.])
# x_in_cam_1 = se3.pose_inv(cam_from_blender_to_opencv(cam1_posemat)) @ x_in_world_hom
# x_in_cam_2 = se3.pose_inv(cam_from_blender_to_opencv(cam2_posemat))@ x_in_world_hom
# x_in_im_1 = intrinsics@x_in_cam_1[:3]
# x_in_im_1 = x_in_im_1/x_in_im_1[2]
# x_in_im_1_rounded = np.round(x_in_im_1).astype('int')[:2]
# x_in_im_2 = intrinsics@x_in_cam_2[:3]
# x_in_im_2 = x_in_im_2/x_in_im_2[2]
# x_in_im_2_rounded = np.round(x_in_im_2).astype('int')[:2]
#
# CameraUtility.add_camera_pose(cam2_posemat, 0)
# correct_camera_posemat = np.array(WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix', local_frame_change=["X", "-Y", "-Z"]))
#
#
#
# data, seg_data, seg_map_cups = render_scene(0, chunk_id, camera_pose, is_bottleneck=True)
#
# cam1_D = data['depth'][0]
#
# CameraUtility.add_camera_pose(cam2_posemat, 0)
# data2, seg_data2, seg_map_cups2 = render_scene(0, chunk_id, camera_pose, is_bottleneck=False)
#
# cam2_D = data2['depth'][0]
#
#
#
# cam1_posemat= cam_from_blender_to_opencv(cam1_posemat)
# cam2_posemat= cam_from_blender_to_opencv(cam2_posemat)
#
#
# height, width = cam1_D.shape
#
# intrinsics_inv = np.linalg.pinv(intrinsics)
# cam1_extrinsics_inv = cam1_posemat  # se3.pose_inv(cam1_posemat ) #
#
# cam2_extrinsics = se3.pose_inv(cam2_posemat)  # cam2_posemat #
#
# # mask_x are the columns (so x in image coords) and mask_y are rows (so y in image coords)
# mask_cols, mask_rows = pixel_coord_np(height, width)
# mask_x = mask_cols.reshape((-1))
# mask_y = mask_rows.reshape((-1))
#
# # Get to world frame coordinates
# d_values = cam1_D[mask_y, mask_x]
# cam1_hom_image_coordinates = np.vstack([mask_x, mask_y, np.ones_like(mask_y)])
# cam1_frame_coords = intrinsics_inv[:3, :3].dot(cam1_hom_image_coordinates) * d_values.flatten()
# cam1_frame_coords = np.vstack([cam1_frame_coords, np.ones((1, cam1_frame_coords.shape[1]))])
# world_frame_coords = cam1_extrinsics_inv.dot(cam1_frame_coords)
#
# # Get to cam2 frame coordinates and then get correspondances to the cam2 image
# cam2_frame_coords = cam2_extrinsics.dot(world_frame_coords)
# cam2_intrinsics = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
# cam2_image_coords = cam2_intrinsics.dot(cam2_frame_coords)
# cam2_image_coords /= cam2_image_coords[-1, :]
# cam2_image_coords = np.round(cam2_image_coords).astype(int)[:2,
#                     :]  # 0'th index is width (columns) and 1st index is height (rows)
#
# cam2_image_correspondances = cam2_image_coords.reshape((2, cam1_D.shape[0], cam1_D.shape[1])).transpose(1, 2,
#                                                                                                         0)  # 0'th channel is width (x, columns)
#
# # Select valid entries that project inside the image
# row0_selection = np.logical_and(cam2_image_coords[0, :] < width, cam2_image_coords[0, :] >= 0)
# row1_selection = np.logical_and(cam2_image_coords[1, :] < height, cam2_image_coords[1, :] >= 0)
# row_selection = np.logical_and(row0_selection, row1_selection)
#
# # Getting the out of frame map
# out_of_frame_map = np.logical_not(row_selection).reshape((cam1_D.shape[0], cam1_D.shape[1]))
#
# # Getting the occlusions map
# occlusions_mask_flattened = -1000 * np.ones((cam1_D.shape[0] * cam1_D.shape[
#     1]))  # Making sure that there are no false positives by seeting a negative buffer array
#
# # For the in-frame correspondances, get the correspondance coordinates
# # cam2_image_coords_selected shows the x,y correspondance of each selected entry of a flattened cam1 image
# # To see which entry in a flattened cam 1 iamge is selected, we can use row_selection_where
# cam2_image_coords_selected = cam2_image_coords[:2, row_selection]
# row_selection_where = np.where(row_selection)
#
# # Get the depth values from cam2 by using the correspondance pixels
# # Put those depth values into the  occlusion mask buffer, note the buffer is in the cam1 frame so we use the row_selection_where
# vals_in_cam2_D = cam2_D[cam2_image_coords_selected[1, :], cam2_image_coords_selected[0, :]]
# occlusions_mask_flattened[row_selection_where] = vals_in_cam2_D
#
# # Reshape and compare what we get to the 3D coordinates of the points in the cam2 frame
# occlusions_mask = occlusions_mask_flattened.reshape(
#     cam1_D.shape)  # We need to get the depth values of cam2 for pixels in cam1 given the correspondances
# cam2_frame_coords_reshaped = cam2_frame_coords[2, :].reshape(
#     cam1_D.shape)  # The 3D coordinates in the cam2 frame for each pixel in cam1
#
# # Allow a tolerance of 1mm for coppelia inaccuracies
# occlusions_mask = np.isclose(occlusions_mask, cam2_frame_coords_reshaped, atol=1.5e-3, rtol=0.).astype('uint8')
#
# # Combining the occlusions map and out of frame map
# semantics_map = 2 * occlusions_mask + out_of_frame_map * -1
# assert not np.isclose(semantics_map, 1.).any()  # sanity check
# semantics_map += 1
#
# print('DONre')
#
#
#

#################################################################


# to get to opencv you have R_blender = R_opencv @ R_opencv_to_blender or R_opencv = R_opencv_to_blender @ R_blender -> it donest matter becasue R_opencv_to_blender^-1 = R_opencv_to_blender
