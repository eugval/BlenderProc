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
from src.Eugene.dataset_utils import run_modules, fix_uv_maps,set_lights, sample_camera_pose,apply_twist,process_bottleneck, aggregate_segmap_values, cam_from_blender_to_opencv, render_scene, superpose_bottleneck,superpose_seg_mask,resize_intrinsics
from src.utility.CameraUtility import CameraUtility
from src.Eugene.my_writer import MyWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.Eugene import se3_tools as se3
from examples.visual_servo.utils.plotting_functions import PlotVisualServo_errors
from dense_correspondence_control.control.visual_servo import get_visual_servoing , get_4d_visual_servoing
from dense_correspondence_control.control.find_correspondences import put_optical_flow_arrows_on_image, scipy_nn_computation, flow_from_correspondences, scipy_nn_computation_with_mask
from dense_correspondence_control.learning.testing.load_networks import load_network
import torch
import cv2
import copy







def propagate(current_rgb,bottleneck_rgb, current_seg_map, bottleneck_seg_map, networks, image_saver, writer, num_frame, use_bottleneck_mask= True) :
    # Propagate
    # Put image in torch format
    rgb_current = torch.tensor(copy.deepcopy(current_rgb)/255.).to(device).float().permute(2,0,1).unsqueeze(0)

    rgb_bottleneck = torch.tensor(copy.deepcopy(bottleneck_rgb)).to(device).float().permute(2,0,1).unsqueeze(0)

    #Get the dense descriptor images
    out_from = networks['model'](rgb_current)
    out_to = networks['model'](rgb_bottleneck)

    seg_out_from = current_seg_map
    seg_out_to = bottleneck_seg_map

    if(not use_bottleneck_mask):
        seg_out_to = np.ones_like(seg_out_to)

    # Get the correspondences
    correspondances, confidences_2d, distances_2d = scipy_nn_computation_with_mask(out_from[0].permute(1,2,0).detach().cpu().numpy(), out_to[0].permute(1,2,0).detach().cpu().numpy(), seg_out_from,seg_out_to)

    image_saver.plot_errors(bottleneck_pose,camera_pose,gt_flattened_correspondences,gt_starting_pixels,save_dir= writer.chunk_tdir.format(chunk_id=writer.chunk_id))

    image_test = put_optical_flow_arrows_on_image(current_rgb, correspondances, mask = seg_out_from.astype('uint8'), threshold=1.0)
    image_test = superpose_bottleneck(image_test/255. ,bottleneck_rgb/255., bottleneck_seg_map)
    image_test = superpose_seg_mask (image_test, seg_out_from.astype('uint8'))


    image_saver.save_image(writer.chunk_tdir.format(chunk_id= writer.chunk_id), 'trajectory', num_frame, image_test)
    image_saver.save_image(writer.chunk_tdir.format(chunk_id= writer.chunk_id), 'prediction_seg_trajectory', num_frame, seg_out_from)

    return correspondances, confidences_2d, distances_2d, seg_out_from






def propagate_with_seg(current_rgb,bottleneck_rgb, bottleneck_seg_map, networks, image_saver, writer, num_frame, use_bottleneck_mask = True) :
    # Propagate
    #Put image in torch format

    rgb_current = torch.tensor(copy.deepcopy(current_rgb)/255.).to(device).float().permute(2,0,1).unsqueeze(0)

    rgb_bottleneck = torch.tensor(copy.deepcopy(bottleneck_rgb)).to(device).float().permute(2,0,1).unsqueeze(0)

    #Get the dense descriptor images
    out_from, seg_out_from, seg_out_from_logits = networks['model'](rgb_current)
    out_to, seg_out_to, seg_out_to_logits = networks['model'](rgb_bottleneck)

    seg_out_from = (seg_out_from.detach().cpu().numpy() > 0.5)[0,0,:,:]
    seg_out_to = (seg_out_to.detach().cpu().numpy() > 0.5)[0,0,:,:]

    if(not use_bottleneck_mask):
        seg_out_to = np.ones_like(seg_out_to)

    # Get the correspondences
    correspondances, confidences_2d, distances_2d = scipy_nn_computation_with_mask(out_from[0].permute(1,2,0).detach().cpu().numpy(), out_to[0].permute(1,2,0).detach().cpu().numpy(), seg_out_from,seg_out_to)

    image_saver.plot_errors(bottleneck_pose,camera_pose,gt_flattened_correspondences,gt_starting_pixels,save_dir= writer.chunk_tdir.format(chunk_id=writer.chunk_id))

    image_test = put_optical_flow_arrows_on_image(current_rgb, correspondances, mask = seg_out_from.astype('uint8'), threshold=1.0)
    image_test = superpose_bottleneck(image_test/255. ,bottleneck_rgb/255., bottleneck_seg_map)
    image_test = superpose_seg_mask (image_test, seg_out_from.astype('uint8'))


    image_saver.save_image(writer.chunk_tdir.format(chunk_id= writer.chunk_id), 'trajectory', num_frame, image_test)
    image_saver.save_image(writer.chunk_tdir.format(chunk_id= writer.chunk_id), 'prediction_seg_trajectory', num_frame, seg_out_from)

    return correspondances, confidences_2d, distances_2d, seg_out_from


temp_save_folder = '/home/eugene/test/'

Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/visual_servo" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')

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

# DEFINE VISUALS
fix_uv_maps()

run_modules(Utility.initialize_modules([{
    "module": "manipulators.MaterialManipulator",
    "config": {
        "selector": {
            "provider": "getter.Material",
            "conditions": {
                "cf_use_materials_of_objects": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_manip_object": True,
                    }
                }
            }
        },
        "cf_set_base_color": [0.15, 0.5, 0.35, 1],
    }
}, ]))

light1, light2 = set_lights(ambient_strength_range=[2.0, 2.0], light_colour_ranges=[[0., 1.0], [0.0, 0.0], [1.0, 1.0]],
           ambient_light_colour=[0.5, 0.5, 0.5, 1.0])

# DEFINE CAMERA
intrinsics = np.array([[355.5556369357639, 0., 127.5],
                     [0,  355.5556369357639, 127.5],
                     [0., 0., 1.]])
CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width =256, image_height=256,clip_start=0.001)



#DEFINE WRITER
writer = MyWriter(Utility.working_dir , 'data/test_trajectory', has_bottleneck=True)
chunk_id = writer.check_and_create_trajectory_folders()
writer.save_camera_intrinsics(CameraUtility.get_intrinsics_as_K_matrix(), chunk_id=chunk_id)
image_saver = PlotVisualServo_errors()

# activate depth rendering
RendererUtility.set_samples(50)
RendererUtility.enable_depth_output()


# Define frames
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1


# Define bottleneck
pos_volume =  ([0.,0.,0.12],[0.,0.,0.18])
orn_volume = ([0.,0.,0.],[0.,0.,0.])
bottleneck_pose, bottleneck_rgb, bottleneck_depth, bottleneck_seg_map =  process_bottleneck(num_frame = 0, chunk_id=chunk_id, pos_volume = pos_volume, orn_volume = orn_volume, manip_object = None, writer=writer)

# randomise_light_position([light1,light2], frame_num = 0) TODO: There is something weird where you can't change light location after rendering

# Define Initial camera pose
cam_pos_delta = np.array([0.03,-0.04, 0.09])
cam_rot_delta = se3.so3_exp(np.array([0.,0.,3.14]).astype(float))
cam_pose_delta = se3.make_pose(cam_pos_delta,cam_rot_delta)
camera_pose = np.matmul(cam_pose_delta,bottleneck_pose)
CameraUtility.add_camera_pose(camera_pose, 0)

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
for num_frame in range(0, num_frames):
    #### GRAB GT DATA #####
    data, seg_data, seg_map_cups = render_scene(num_frame, chunk_id, camera_pose, is_bottleneck=False, writer=writer)

    current_rgb = cv2.resize(data['colors'][0], (128,128))
    current_depth = cv2.resize(data['depth'][0], (128,128))
    seg_map_cups = cv2.resize(seg_map_cups,(128,128))
    bottleneck_rgb = cv2.resize( bottleneck_rgb, (128,128))
    bottleneck_depth = cv2.resize( bottleneck_depth, (128,128))
    bottleneck_seg_map = cv2.resize( bottleneck_seg_map, (128,128))


    resize_ratio = (128/256, 128/256)
    resized_intrinsics = resize_intrinsics(CameraUtility.get_intrinsics_as_K_matrix(), resize_ratio)

    # current_rgb =data['colors'][0]
    # current_depth = data['depth'][0]
    # seg_map_cups = seg_map_cups
    # bottleneck_rgb = bottleneck_rgb
    # bottleneck_depth = bottleneck_depth
    # bottleneck_seg_map = bottleneck_seg_map
    #
    # resize_ratio = (128 / 256, 128 / 256)
    # resized_intrinsics = CameraUtility.get_intrinsics_as_K_matrix()



    gt_cam2_image_correspondances, gt_semantics_map = writer.get_correspondance_map(resized_intrinsics,
                                                                             cam_from_blender_to_opencv(camera_pose),
                                                                             cam_from_blender_to_opencv(
                                                                                 bottleneck_pose),
                                                                             current_depth, bottleneck_depth)

    gt_visible_object_pixels = seg_map_cups * (gt_semantics_map == 3).astype('uint8')
    gt_flattened_correspondences = gt_cam2_image_correspondances[gt_visible_object_pixels > 0.]
    gt_flattened_depth = current_depth[gt_visible_object_pixels > 0.]
    gt_flattened_bottleneck_depth = bottleneck_depth[gt_visible_object_pixels > 0.]
    gt_visible_object_pixels_selected = np.where(gt_visible_object_pixels > 0.)
    gt_starting_pixels = np.stack([gt_visible_object_pixels_selected[1], gt_visible_object_pixels_selected[0]], axis=1)


    ##### Grab NN data ####
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Load network
    networks, _ = load_network(
        '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/models/DoNs',
        60, device=device)  # DoN_with_seg_balanced',395,

    # correspondances, confidences_2d, distances_2d, seg_out_from = propagate_with_seg(current_rgb, bottleneck_rgb, bottleneck_seg_map,  networks,
    #                                                                                  image_saver, writer, num_frame, use_bottleneck_mask= True)
    correspondances, confidences_2d, distances_2d, seg_out_from = propagate(current_rgb, bottleneck_rgb,seg_map_cups, bottleneck_seg_map,  networks,
                                                                                     image_saver, writer, num_frame, use_bottleneck_mask= False)

    #### Get Twist ####
    visible_object_pixels = seg_out_from.astype('uint8')
    flattened_correspondences = correspondances[visible_object_pixels > 0.]
    visible_object_pixels_selected = np.where(visible_object_pixels > 0.)
    starting_pixels = np.stack([visible_object_pixels_selected[1], visible_object_pixels_selected[0]], axis=1)

    twist = get_4d_visual_servoing(source_points=starting_pixels, target_points=flattened_correspondences,
                                   intrinsics=resized_intrinsics, s_gain=0.01, xy_gain=0.01, theta_gain=0.35)




    #Apply twist and set new camera pose
    camera_pose_initial_opencv = cam_from_blender_to_opencv(camera_pose)
    step = 1.0
    camera_pose_final_opencv = apply_twist(camera_pose_initial_opencv, twist, step)
    camera_pose = cam_from_blender_to_opencv(camera_pose_final_opencv)

    CameraUtility.add_camera_pose(camera_pose, num_frame)



    # Advance Frame
    # randomise_light_position([light1,light2], frame_num = num_frame)
    bpy.context.scene.frame_start = num_frame
    bpy.context.scene.frame_end = num_frame + 1

map)