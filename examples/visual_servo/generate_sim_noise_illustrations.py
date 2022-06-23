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
from src.Eugene.dataset_utils import run_modules, fix_uv_maps,set_lights, min_max_stardardisation_to_01, perlin_bitmap_deformation,sample_camera_pose,sample_camera_pose_delta,randomise_light_position_colour_and_energy,apply_twist, aggregate_segmap_values, cam_from_blender_to_opencv, render_scene, superpose_bottleneck,superpose_seg_mask,resize_intrinsics,light_setting_selection, deform_segmentation,deform_segmentation_bottleneck,  generate_perlin_noise,generate_random_mis_segmentation_map
from src.utility.CameraUtility import CameraUtility
from src.Eugene.my_writer import MyWriter
from src.Eugene.new_writer import MyNewWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.Eugene import se3_tools as se3
from src.Eugene.module_run_blocks import load_modelnet, randomise_manip_object_colour, randomise_room,add_distractors
from examples.visual_servo.utils.plotting_functions import PlotVisualServo_errors
from dense_correspondence_control.control.visual_servo import get_visual_servoing , get_4d_visual_servo_with_segs
from dense_correspondence_control.control.segmentation_control import SegmentationController
from dense_correspondence_control.control.manual_control import ManualController
from dense_correspondence_control.control.direct_regression_control import RegressionController
from dense_correspondence_control.utils.image_utils import resize_with_PIL, resize_with_cv2
from dense_correspondence_control.utils.logger import write_dict_to_file

import torch
import cv2
import copy
import pickle
import time
from src.utility.MeshObjectUtility import MeshObject
from src.utility.filter.Filter import Filter
import random
import collections
import argparse


temp_save_folder = '/home/eugene/test/'


def process_bottleneck(pos_volume = None, orn_volume = None, object_name = None):
    # Place camera at bottleneck
    # Get object of interest

    T_wc = sample_camera_pose(pos_volume, orn_volume,
                              pre_selected_visible_objects=Filter.by_cp(
                                  elements=MeshObject.convert_to_meshes(bpy.data.objects), cp_name='manip_object',
                                  value=True),
                                fully_visible=True, obj_name_check=object_name, relative_to_top=False)

    CameraUtility.add_camera_pose(T_wc, frame=0)


    # render
    data, seg_data, seg_map_obj = render_scene(activate_renderer = True)

    # return bottleneck stuff
    return T_wc, data['colors'][0], data['depth'][0], seg_map_obj


def current_to_target_seg_illustration(current_seg, target_seg, current_rgb):
    alpha = 0.4
    new_image = copy.copy(current_rgb)/255.
    new_image =    np.clip(
                    np.expand_dims(target_seg, axis=-1) * alpha * [0., 1., 0.] + \
                    np.expand_dims(target_seg, axis=-1) * new_image * (1 - alpha) + \
                    (1.0- np.expand_dims(target_seg, axis=-1)) * new_image ,
                    a_min=0., a_max=1.)


    new_image = np.clip(
        np.expand_dims(current_seg, axis=-1) * alpha * [1., 0., 0.] + \
        np.expand_dims(current_seg, axis=-1) * new_image * (1 - alpha) +\
        (1.0-  np.expand_dims(current_seg, axis=-1)) * new_image,
        a_min=0., a_max=1.)

    return new_image



Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/visual_servo" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')

# LOAD THE OBJECT INTO THE SCENE
load_modelnet()
object_name = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True)[0].get_name()

# Randomise visuals
fix_uv_maps()
randomise_manip_object_colour([0.2,0.8])
randomise_room()

ambient_strength_range,light_location_volume,number_of_lights,energy_range = light_setting_selection('generic_shadow')
created_lights, light_colour = set_lights(
    ambient_strength_range  = ambient_strength_range,
    light_colour_ranges = [[0., 0.2], [0.0, 1.0], [0.85, 1.0]],
    ambient_light_colour =  [0.5, 0.5, 0.5, 1.0],
    light_location_volume=light_location_volume,
    number_of_lights=number_of_lights,
    energy_range =energy_range)


# DEFINE CAMERA
intrinsics = np.array([[128., 0.,63.5],
                       [0.,126.7, 63.5 ],
                       [0.,0.,1.]])
CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 128, image_height = 128,clip_start=0.001)

#DEFINE WRITER
writer = MyNewWriter(Utility.working_dir ,'data/illustrations/')
chunk_id = writer.check_and_create_trajectory_folders()
# image_saver = PlotVisualServo_errors()


# Define frames
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1


current_mis_segmentation_map = None


bottleneck_perlin_noise = generate_perlin_noise((64,64))
current_perlin_noise = generate_perlin_noise((64,64))


current_mis_segmentation_map_1 = generate_random_mis_segmentation_map((64,64), num_segs = 1,
                                                                    seg_size = np.random.randint(4,12), seed = 0)

current_mis_segmentation_map_2 = generate_random_mis_segmentation_map((64,64), num_segs = 2,
                                                                    seg_size = np.random.randint(4,12), seed = 0)

current_mis_segmentation_map_3 = generate_random_mis_segmentation_map((64,64), num_segs = 3,
                                                                    seg_size = np.random.randint(4,12), seed = 0)




# PLACE CAMERA
pos_volume = ([-0.11, -0.11, -0.06], [0.11, 0.11, 0.30])
orn_volume = ([0., 0., -np.pi/2], [0., 0., np.pi/2])
camera_pose =sample_camera_pose(pos_volume, orn_volume,
                              pre_selected_visible_objects=Filter.by_cp(
                                  elements=MeshObject.convert_to_meshes(bpy.data.objects), cp_name='manip_object',
                                  value=True),
                                fully_visible=True, obj_name_check=object_name, relative_to_top=False)
CameraUtility.add_camera_pose(camera_pose, 0)

#### GRAB GT DATA #####
data, seg_data, seg_map_obj = render_scene()
current_image = data['colors'][0]
current_image = resize_with_PIL(current_image,64)
seg_map_obj = resize_with_cv2(seg_map_obj,64,64)
deformed_current_seg_map_perlin = deform_segmentation(seg_map_obj,'perlin',
                                               perlin_noise_parameters=current_perlin_noise,
                                               extra_segmentation=current_mis_segmentation_map_1,
                                               seed=0)

deformed_current_seg_map_extra_1 = deform_segmentation(seg_map_obj,'extra_1',
                                               perlin_noise_parameters=current_perlin_noise,
                                               extra_segmentation=current_mis_segmentation_map_1,
                                               seed=0)

deformed_current_seg_map_extra_2 = deform_segmentation(seg_map_obj,'extra_2',
                                               perlin_noise_parameters=current_perlin_noise,
                                               extra_segmentation=current_mis_segmentation_map_2,
                                               seed=0)

deformed_current_seg_map_extra_3 = deform_segmentation(seg_map_obj, 'extra_3',
                                                       perlin_noise_parameters=current_perlin_noise,
                                                       extra_segmentation=current_mis_segmentation_map_3,
                                                       seed =0)




plt.imsave(writer.make_path('current_seg',0,'.png', chunk_id), seg_map_obj)
plt.imsave(writer.make_path('current_seg',1, '.png', chunk_id), deformed_current_seg_map_perlin)
plt.imsave(writer.make_path('current_seg',2, '.png', chunk_id), deformed_current_seg_map_extra_1)
plt.imsave(writer.make_path('current_seg',3, '.png', chunk_id), deformed_current_seg_map_extra_2)
plt.imsave(writer.make_path('current_seg',4, '.png', chunk_id), deformed_current_seg_map_extra_3)
plt.imsave(writer.make_path('current_rgb',5, '.png', chunk_id), current_image)


