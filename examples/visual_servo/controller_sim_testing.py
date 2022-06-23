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

parser = argparse.ArgumentParser()
parser.add_argument('controller_type', help="One of [ segmentation, ]", default = 'segmentation')
parser.add_argument('environment_type', help="One of [ perfect, ]", default = 'perfect')
parser.add_argument('seed', help="The random seed to apply", default = '0')
parser.add_argument('gains_multiplier',help='modify the gains for gains ablation', default = 1.)
args = parser.parse_args()

controller_type = args.controller_type
environment_type = args.environment_type
gains_multiplier = 5154
seed =int(args.seed)
np.random.seed(seed)
random.seed(seed)
#
# #
# seed = 0
# controller_type = 'segmentation'
# environment_type = 'perfect'
# np.random.seed(seed)
# random.seed(seed)

xy_gain = 0.07#*gains_multiplier
theta_gain = 0.4#*gains_multiplier
s_gain = 0.015#*gains_multiplier
debug = False

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
writer = MyNewWriter(Utility.working_dir ,'data/controller_testing/{}/{}/{}'.format(gains_multiplier, controller_type,environment_type))
chunk_id = writer.check_and_create_trajectory_folders()
# image_saver = PlotVisualServo_errors()


# Define frames
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1


# Grab the bottleneck
bottleneck_perlin_noise = None
current_perlin_noise = None
current_mis_segmentation_map = None

print(controller_type)
if(('perlin' in environment_type) or ('extra' in environment_type)):
    print('in here')
    bottleneck_perlin_noise = generate_perlin_noise((64,64),seed)
    current_perlin_noise = generate_perlin_noise((64,64),seed)

if('extra' in environment_type):
    num_segs = int(environment_type.split('_')[-1])
    current_mis_segmentation_map = generate_random_mis_segmentation_map((64,64), num_segs = num_segs,
                                                                        seg_size = np.random.randint(4,12), seed = 0)



bottleneck_pose, bottleneck_rgb, \
bottleneck_depth, bottleneck_seg_map =  process_bottleneck(pos_volume = ([-0.0, -0.0, 0.2],[0.0, 0.0, 0.2]),
                                                           orn_volume = ([0.,0.,0],[0.,0.,0]),
                                                           object_name=object_name)
bottleneck_rgb = resize_with_PIL(bottleneck_rgb,64)
bottleneck_seg_map = resize_with_cv2(bottleneck_seg_map,64,64)



deformed_bottleneck_seg_map = deform_segmentation_bottleneck(bottleneck_seg_map,environment_type, perlin_noise_parameters=bottleneck_perlin_noise, seed=seed )

if(debug):
    writer.save_rgb(bottleneck_rgb, writer.make_path('bot_rgb',0,'.png',chunk_id))
    plt.imsave(writer.make_path('bot_seg',0,'.png',chunk_id), deformed_bottleneck_seg_map )

##### ADD DISTRACTORS and change lights
add_distractors(number_of_distractor = np.random.randint(0,15),
                distractor_max_dist =  np.random.uniform(0.1, 0.3),
                distractor_offset_x_dist = np.random.uniform(-0.1,0.1),
                distractor_offset_y_dist = np.random.uniform(-0.1,0.1)
                )

if (environment_type == 'lights'):
    for light in created_lights:
        light.delete()

    ambient_strength_range, light_location_volume, number_of_lights, energy_range = light_setting_selection(
        'generic_shadow')
    created_lights, light_colour = set_lights(
        ambient_strength_range=ambient_strength_range,
        light_colour_ranges=[[0., 0.2], [0.0, 1.0], [0.85, 1.0]],
        ambient_light_colour=[0.5, 0.5, 0.5, 1.0],
        light_location_volume=light_location_volume,
        number_of_lights=number_of_lights,
        energy_range=energy_range)



# PLACE CAMERA
pos_volume = ([-0.11, -0.11, -0.06], [0.11, 0.11, 0.26])
orn_volume = ([0., 0., -np.pi/2], [0., 0., np.pi/2])
camera_pose = sample_camera_pose_delta(copy.deepcopy(bottleneck_pose), pos_volume, orn_volume,
                                  pre_selected_visible_objects = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,
                                                                              cp_name='manip_object',value=True),
                                  fully_visible = True,
                                  obj_name_check =object_name,
                                 )
CameraUtility.add_camera_pose(camera_pose, 0)



# INITIALISE CONTROLLER
if (controller_type == 'segmentation'):
    controller = ManualController(xy_pix_threshold = 0.01,
                                        s_theshold = 0.01,
                                        angular_threshold = 1.,)
    manual_score_function ='iou'
elif (controller_type == 'segmentation_rgb'):
    controller = ManualController(xy_pix_threshold = 0.01,
                                        s_theshold = 0.01,
                                        angular_threshold = 1.,)
    manual_score_function ='l1'

elif (controller_type == 'regression_noise'):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dr_nets = '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/models/direct_regression'
    angle_network_type = 'sincos'
    angle_network_path = os.path.join(dr_nets, 'z_angle_sin_cos_noise')
    angle_network_checkpoint = 295

    xy_network_path = os.path.join(dr_nets, 'dr_xy_correct_scale_noise')
    xy_network_checkpoint = 295

    s_network_path = os.path.join(dr_nets, 'dr_scales_capped_scale_loss_noise')
    s_network_checkpoint = 150

    controller = RegressionController(
        device=device,
        angle_network_path=angle_network_path,
        angle_network_checkpoint=angle_network_checkpoint,
        angle_network_type=angle_network_type,
        xy_network_path=xy_network_path,
        xy_network_checkpoint=xy_network_checkpoint,
        s_network_path=s_network_path,
        s_network_checkpoint=s_network_checkpoint,
        sxy_network_path=None,
        sxy_network_checkpoint=None,
        xy_pix_threshold=0.02,
        s_theshold=0.01,
        angular_threshold= 1.5, )

elif (controller_type == 'regression'):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dr_nets = '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/models/direct_regression'
    angle_network_type = 'sincos'
    angle_network_path = os.path.join(dr_nets, 'dr_z_angle_sin_cos')
    angle_network_checkpoint = 349

    xy_network_path = os.path.join(dr_nets, 'dr_xy_correct_scale')
    xy_network_checkpoint = 349

    s_network_path = os.path.join(dr_nets, 'dr_scales_capped_scale_loss')
    s_network_checkpoint = 245

    controller = RegressionController(
                device=device,
                 angle_network_path=angle_network_path,
                 angle_network_checkpoint=angle_network_checkpoint,
                 angle_network_type=angle_network_type,
                 xy_network_path=xy_network_path,
                 xy_network_checkpoint=xy_network_checkpoint,
                 s_network_path=s_network_path,
                 s_network_checkpoint=s_network_checkpoint,
                 sxy_network_path=None,
                 sxy_network_checkpoint=None,
                 xy_pix_threshold=0.02,
                 s_theshold= 0.01,
                 angular_threshold=1.5,)

elif controller_type == 'flow':
    pass

else:
    raise NotImplementedError()






# Analysis metrics
camera_pos_errors = []
camera_pos_errors_xy = []
camera_pos_errors_z = []
rot_errors = []

max_frames = 1000
mean_angle = None
converged = False
count = 0
for num_frame in range(0, max_frames):
    if(converged):
        break
    #### GRAB GT DATA #####
    data, seg_data, seg_map_obj = render_scene()
    current_image = data['colors'][0]
    current_image = resize_with_PIL(current_image,64)
    seg_map_obj = resize_with_cv2(seg_map_obj,64,64)
    deformed_current_seg_map = deform_segmentation(seg_map_obj,environment_type,
                                                   perlin_noise_parameters=current_perlin_noise,
                                                   extra_segmentation=current_mis_segmentation_map,
                                                   seed=seed)


    #### Get Twist ####
    a = time.time()
    if ('segmentation' in controller_type):
        twist, info = controller.get_4d_visual_servo(  current_rgb=current_image, bottleneck_rgb=bottleneck_rgb,
                                                    current_seg=deformed_current_seg_map, target_seg=deformed_bottleneck_seg_map,
                                                    s_gain=s_gain, xy_gain =xy_gain, theta_gain=theta_gain, previous_angle_delta = mean_angle,
                                                     score_fnct = manual_score_function )
    elif( 'regression' in controller_type):
        twist, info = controller.get_4d_visual_servo(current_rgb = current_image,
                                                               bottleneck_rgb = bottleneck_rgb,
                                                               current_seg = deformed_current_seg_map,
                                                               target_seg = deformed_bottleneck_seg_map,
                                                               xy_gain =  xy_gain,
                                                               s_gain =s_gain,
                                                               theta_gain = theta_gain)


    # if info['angle_delta'] is not None:
    #     mean_angle = info['angle_delta']
    #     mean_angle_unfiltered = info['angle_delta_unfiltered']
    #     print('angle : {} . Unfiltered : {}'.format(mean_angle, mean_angle_unfiltered))
    #     print(twist)


    #Apply twist and set new camera pose
    camera_pose_initial_opencv = cam_from_blender_to_opencv(camera_pose)
    camera_pose_final_opencv = apply_twist(camera_pose_initial_opencv, twist)
    camera_pose = cam_from_blender_to_opencv(camera_pose_final_opencv)

    CameraUtility.add_camera_pose(camera_pose, num_frame)

    ### stopping criterion
    if(info['stop']):
        print('CONVERGED')
        converged = True

    #### Plot every 10 frames
    if((num_frame%3== 0 or converged) and debug):

        if(converged):
            data, seg_data, seg_map_obj = render_scene()
            current_image = data['colors'][0]
            current_image = resize_with_PIL(current_image, 64)
            seg_map_obj = resize_with_cv2(seg_map_obj, 64, 64)
            deformed_current_seg_map = deform_segmentation(seg_map_obj,environment_type,
                                                           perlin_noise_parameters=current_perlin_noise,
                                                           extra_segmentation = current_mis_segmentation_map,
                                                           seed=seed)

        writer.save_rgb(current_image, writer.make_path('current_rgb', num_frame, '.png', chunk_id))
        plt.imsave(writer.make_path('current_seg', num_frame, '.png', chunk_id), deformed_current_seg_map)

        current_to_target_illustration = current_to_target_seg_illustration(deformed_current_seg_map, deformed_bottleneck_seg_map,
                                                                            current_image)
        c_t_t_illustration_path = writer.make_path('trajectory', num_frame, '.png', chunk_id)
        writer.save_rgb((current_to_target_illustration * 255).astype('uint8'), c_t_t_illustration_path)

        delt = bottleneck_pose[:3, 3] - camera_pose[:3, 3]
        camera_pos_errors.append(np.linalg.norm(delt))
        camera_pos_errors_xy.append(np.linalg.norm(delt[:2]))
        camera_pos_errors_z.append(np.abs(delt[2]))
        plt.plot(camera_pos_errors, label='norm')
        plt.plot(camera_pos_errors_xy, label='xy')
        plt.plot(camera_pos_errors_z, label='z')
        plt.legend()
        plt.ylim((np.min(camera_pos_errors + camera_pos_errors_xy + camera_pos_errors_z),
                  np.max(camera_pos_errors + camera_pos_errors_xy + camera_pos_errors_z)))
        plt.savefig(os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'plot_position_error.png'))
        plt.close()

        camera_rot_delta = bottleneck_pose[:3, :3] @ camera_pose[:3, :3].T
        camera_rotvec_delta = se3.rot2euler('XYZ', camera_rot_delta)
        rot_error = np.abs(camera_rotvec_delta[2] * 180. / np.pi)
        rot_errors.append(rot_error)
        plt.plot(rot_errors)
        plt.ylim((np.min(rot_errors), np.max(rot_errors)))
        plt.savefig(os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'plot_rotation_error.png'))
        plt.close()


    # Advance Frame

    bpy.context.scene.frame_start = num_frame
    bpy.context.scene.frame_end = num_frame + 1
    count +=1

result = {
    'pos_error': camera_pos_errors[-1],
     'pos_error_xy': camera_pos_errors_xy[-1],
    'pos_error_z' : camera_pos_errors_z[-1],
    'angle_error' : rot_errors[-1],
    'converged': int(converged),
    'steps_to_convergence': count,
}

writer.save_object_info(result,os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'result.pckl'))
write_dict_to_file(result,
                   path=os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'result.txt'),
                   title='Results',
                   append=False,
                   newline=True)

np.save(os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'done.npy'), np.array([1]))
