from src.utility.SetupUtility import SetupUtility
argv = SetupUtility.setup([   ]) #'pydevd-pycharm~=193.6911.2'
from src.utility.LockerUtility import Locker

import matplotlib.pyplot as plt

import os
import bpy
import sys
import numpy as np
from src.utility.Utility import Utility
from src.utility.Initializer import Initializer
from src.utility.loader.BlendLoader import BlendLoader
from src.Eugene.dataset_utils import run_modules, fix_uv_maps,set_lights, sample_camera_pose,randomise_light_position_colour_and_energy,apply_twist, aggregate_segmap_values, cam_from_blender_to_opencv, render_scene, superpose_bottleneck,superpose_seg_mask,resize_intrinsics,light_setting_selection
from src.utility.CameraUtility import CameraUtility
from src.Eugene.my_writer import MyWriter
from src.Eugene.new_writer import MyNewWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.Eugene import se3_tools as se3
from examples.visual_servo.utils.plotting_functions import PlotVisualServo_errors
from dense_correspondence_control.control.visual_servo import get_visual_servoing , get_4d_visual_servo_with_segs
from dense_correspondence_control.control.segmentation_control import SegmentationController
import torch
import cv2
import copy
import pickle
import time

temp_save_folder = '/home/eugene/test/'

save_folder_name = 'segmentation_tests'

def min_max_stardardisation_to_01(image):
    image_min = np.min(image)
    image_max = np.max(image)
    range = (image_max-image_min)

    return (image-image_min)/range


def process_bottleneck(pos_volume = None, orn_volume = None):
    # Place camera at bottleneck
    # Get object of interest

    pos = np.random.uniform(*pos_volume)
    orn = np.random.uniform(*orn_volume)
    rot = se3.euler2rot('XYZ', orn)

    pose = np.eye(4)
    pose[:3, 3] = pos
    pose[:3, :3] = rot

    CameraUtility.add_camera_pose(pose, 0)


    # render
    data, seg_data, seg_map_obj = render_scene(activate_renderer = True)

    # return bottleneck stuff
    return pose, data['colors'][0], data['depth'][0], seg_map_obj




def superpose_segs_trajectory( current_seg, target_seg, current_rgb):
    alpha = 0.3
    new_image = copy.copy(current_rgb)/255.
    new_image =    np.clip(np.expand_dims(target_seg, axis=-1) * alpha * [0., 1., 0.] + new_image * (1 - alpha), a_min=0., a_max=1.)
    new_image = np.clip(np.expand_dims(current_seg, axis=-1) * alpha * [1., 0., 0.] + new_image * (1 - alpha), a_min=0., a_max=1.)
    return new_image



Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/visual_servo" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')

# LOAD THE OBJECT INTO THE SCENE
object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ModelNetLoader",
        "config": {
          "data_path": '../../resources/other_objects', #other_objects ModelNet40
          "model_name": "cube", #cube cup_0100
           # "scale_path": "./per_category_scale.json",
            "manual_scale": 0.15,
          "train": True,
          "samples" : 1,
          "replacement": False,
          "add_properties": {
            "cp_physics": True,
            "cp_manip_object":True,
              "cp_distractor": False,
          },
          "cf_set_shading": "SMOOTH"
        }
      },

])

run_modules(object_loading_module)

# DEFINE VISUALS
fix_uv_maps()

cloth_colours_path  = '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/for_colour_extraction/segcolours.pckl'
cloth_colours = pickle.load(open(cloth_colours_path,'rb'))
cloth_colours = np.mean(np.array(cloth_colours), axis =0)/255.

# Randomise the object
run_modules(Utility.initialize_modules([{
        "module": "manipulators.MaterialRandomiser",
        "config": {
            "selector": {
                "provider": "getter.Material",
                "conditions": {
                    "cf_use_materials_of_objects": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "type": "MESH",
                            "cp_manip_object": True,
                        }
                    }
                }
            },
            "mode":"once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": [ "base_color",  "roughness", "metallic", "specular", "anisotropic", "sheen",
                                        "clearcoat"],
            "randomisation_probabilities": [1.0, 0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
            "metallic_min": 0.5,
            "anisotropic_min": 0.5,
            "clearcoat_min": 0.5,
            "clearcoat_roughness_min": 0.5,
            "sheen_min": 0.5,
            "keep_base_color": True,
            # "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "reference_texture_prob": 0.,
            "store_reference_texture": True,
            "randomisation_types": ["monochrome_random",  "image_random"], #  "image_random"
            "randomisation_type_probabilities": [0.5,0.5],# [0.2,0.8][0.7,0.3],[0.5,0.5]
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))





### Randomise the walls

run_modules(Utility.initialize_modules([

    {
        "module": "manipulators.MaterialManipulator",
        "config": {
            "selector": {
                "provider": "getter.Material",
                "conditions": {
                    "cf_use_materials_of_objects": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "cp_manip_object": False,
                        }
                    }
                }
            },
            "cf_set_base_color": [cloth_colours[2], cloth_colours[1], cloth_colours[0], 1.0],  # cloth colours is brg
        }
    },

    {
        "module": "manipulators.MaterialRandomiser",
        "config": {
            "selector": {
                "provider": "getter.Material",
                "conditions": {
                    "cf_use_materials_of_objects": {
                        "provider": "getter.Entity",
                        "conditions": {
                            "type": "MESH",
                            "cp_manip_object": False,
                        }
                    }
                }
            },
            "mode": "once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": ["roughness", "metallic", "specular", "anisotropic", "sheen",
                                        "clearcoat"],
            "randomisation_probabilities": [0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
            "metallic_min": 0.5,
            "anisotropic_min": 0.5,
            "clearcoat_min": 0.5,
            "clearcoat_roughness_min": 0.5,
            "sheen_min": 0.5,
            "keep_base_color": True,
            # "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "randomisation_types": ["monochrome_random"],
            "randomisation_type_probabilities": [1.0],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    }
]))




ambient_strength_range,light_location_volume,number_of_lights,energy_range = light_setting_selection('generic_shadow')
light_colour_ranges = [[0., 0.25], [0.0, 1.0], [0.8, 1.0]]
ambient_light_colour = [0.5, 0.5, 0.5, 1.0]
created_lights, light_colour = set_lights(
    ambient_strength_range  = ambient_strength_range,
    light_colour_ranges = light_colour_ranges,
    ambient_light_colour = ambient_light_colour,
    light_location_volume=light_location_volume,
    number_of_lights=number_of_lights,
    energy_range =energy_range)

# DEFINE CAMERA
intrinsics = np.array([[128., 0.,63.5],
                       [0.,126.7, 63.5 ],
                       [0.,0.,1.]])
CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 128, image_height = 128,clip_start=0.001)

#DEFINE WRITER
writer = MyNewWriter(Utility.working_dir ,'data/controller_testing/')
chunk_id = writer.check_and_create_trajectory_folders()
intrinsics_path = os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'intrinsics.npy')
writer.save_camera_intrinsics(CameraUtility.get_intrinsics_as_K_matrix(), intrinsics_path)
image_saver = PlotVisualServo_errors()
poses_path = os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'poses.pckl')

# activate depth rendering



# Define frames
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1


# Define bottleneck
pos_volume =  ([0.,0.,0.2],[0.,0.,0.2])
orn_volume = ([0.,0.,0.],[0.,0.,0.])
bottleneck_pose, bottleneck_rgb, bottleneck_depth, bottleneck_seg_map =  process_bottleneck(pos_volume = pos_volume, orn_volume = orn_volume)

writer.save_camera_pose(bottleneck_pose, poses_path, 'bottleneck' )
writer.save_rgb(bottleneck_rgb, writer.make_path('bot_rgb',0,'.png',chunk_id))
cv2.imwrite(writer.make_path('bot_seg',0,'.png',chunk_id), bottleneck_seg_map )


# Define Initial camera pose
cam_pos_delta = np.array([-0.0,-0.0, 0.])
cam_rot_delta = se3.so3_exp(np.array([0.,0.,3.14/4]).astype(float))
cam_pose_delta = se3.make_pose(cam_pos_delta,cam_rot_delta)
camera_pose = np.matmul(cam_pose_delta,bottleneck_pose)
CameraUtility.add_camera_pose(camera_pose, 0)



# Initialise controller
controller = SegmentationController()



# Analysis metrics
camera_pos_errors =[]
camera_pos_errors_x =[]
camera_pos_errors_y =[]
camera_pos_errors_z =[]
average_flows_x = []
average_flows_y = []
average_flows = []
rot_errors=[]
v_errors = []

num_frames = 400
mean_angle = None
for num_frame in range(0, num_frames):
    #### GRAB GT DATA #####

    data, seg_data, seg_map_obj = render_scene()
    writer.save_camera_pose(camera_pose, poses_path, '{:06d}'.format(num_frame))
    writer.save_rgb( data['colors'][0], writer.make_path('current_rgb', num_frame, '.png', chunk_id))
    cv2.imwrite(writer.make_path('current_seg',0,'.png',chunk_id), seg_map_obj )

    traj_image = superpose_segs_trajectory(seg_map_obj, bottleneck_seg_map,data['colors'][0] )
    traj_image_path = writer.make_path('trajectory', num_frame,'.png', chunk_id)
    writer.save_rgb((traj_image*255).astype('uint8'), traj_image_path,)

    delt = bottleneck_pose[:3, 3] - camera_pose[:3, 3]
    camera_pos_error = np.linalg.norm(delt)
    camera_pos_error_x = np.abs(delt[0])
    camera_pos_error_y = np.abs(delt[1])
    camera_pos_error_z = np.abs(delt[2])
    camera_pos_errors.append(camera_pos_error)
    camera_pos_errors_y.append(camera_pos_error_y)
    camera_pos_errors_x.append(camera_pos_error_x)
    camera_pos_errors_z.append(camera_pos_error_z)
    plt.plot(camera_pos_errors, label='norm')
    plt.plot(camera_pos_errors_x, label='x')
    plt.plot(camera_pos_errors_y, label='y')
    plt.plot(camera_pos_errors_z, label='z')
    plt.legend()
    plt.ylim((np.min(camera_pos_errors + camera_pos_errors_x + camera_pos_errors_y + camera_pos_errors_z),
              np.max(camera_pos_errors + camera_pos_errors_x + camera_pos_errors_y + camera_pos_errors_z)))
    plt.savefig(os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'plot_position_error.png'))
    plt.close()

    camera_rot_delta = bottleneck_pose[:3, :3] @ camera_pose[:3, :3].T
    camera_rotvec_delta = se3.rot2rotvec(camera_rot_delta)
    rot_error = np.linalg.norm(camera_rotvec_delta)
    rot_errors.append(rot_error)
    plot_rot_errors = np.array(rot_errors)*180./np.pi
    plt.plot(plot_rot_errors)
    plt.ylim((np.min(plot_rot_errors), np.max(plot_rot_errors)))
    plt.savefig(os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'plot_rotation_error.png') )
    plt.close()

    #### Get Twist ####(seg_current, seg_target, xy_gain, s_gain,   theta_gain ) seg_current, seg_target, xy_gain, s_gain, theta_gain, mean_angle = None
    a = time.time()
    twist, info = controller.get_4d_visual_servo_with_segs(
                                                seg_current=seg_map_obj, seg_target=bottleneck_seg_map,
                                                s_gain=0.05, xy_gain = 0.05, theta_gain=0.15, previous_angle_delta = mean_angle
                                                     )

    mean_angle = info['angle_delta']
    mean_angle_uncertainty = info['angle_delta_uncertainty']
    mean_angle_unfiltered = info['angle_delta_unfiltered']
    print('angle : {} +- {} . Unfiltered : {}'.format(mean_angle, mean_angle_uncertainty, mean_angle_unfiltered))
    print(twist)




    #Apply twist and set new camera pose
    camera_pose_initial_opencv = cam_from_blender_to_opencv(camera_pose)
    step = 1.0
    camera_pose_final_opencv = apply_twist(camera_pose_initial_opencv, twist, step)
    camera_pose = cam_from_blender_to_opencv(camera_pose_final_opencv)

    CameraUtility.add_camera_pose(camera_pose, num_frame)


    # Simulate the computatio of the mean_angle

    if(mean_angle is not None):
        initial_angle = se3.rot2euler('XYZ', camera_pose_initial_opencv[:3,:3])[2]
        target_angle = initial_angle + mean_angle

        mean_angle = target_angle - se3.rot2euler('XYZ', camera_pose_final_opencv[:3,:3])[2]




    # Advance Frame
    #TODO: cannot randomise lights after renderging has been done.
    # randomise_light_position_colour_and_energy([light1, light2], colour_change_percent=0.5, energy_range=[50, 350],
    #                                            frame_num=num_frame)
    bpy.context.scene.frame_start = num_frame
    bpy.context.scene.frame_end = num_frame + 1


    # plt.imsave(temp_save_folder+'/d2.png', current_depth)
    # plt.imsave(temp_save_folder+'/db2.png', bottleneck_depth)
    # plt.imsave(temp_save_folder+'/vis2.png', gt_visible_object_pixels)
    # plt.imsave(temp_save_folder + '/seg2.png', seg_map_cups)
    # plt.imsave(temp_save_folder + '/segb2.png', bottleneck_seg_map)
    # plt.imsave(temp_save_folder + '/sem2.png', gt_semantics_map)