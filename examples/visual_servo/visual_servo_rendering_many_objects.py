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
from src.Eugene.dataset_utils import run_modules, fix_uv_maps,set_lights, sample_camera_pose,randomise_light_position_colour_and_energy,apply_twist,process_bottleneck, aggregate_segmap_values, cam_from_blender_to_opencv, render_scene, superpose_bottleneck,superpose_seg_mask,resize_intrinsics
from src.utility.CameraUtility import CameraUtility
from src.Eugene.my_writer import MyWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.Eugene import se3_tools as se3
from examples.visual_servo.utils.plotting_functions import PlotVisualServo_errors
from dense_correspondence_control.control.visual_servo import get_visual_servoing , get_4d_visual_servoing
from dense_correspondence_control.control.find_correspondences import put_optical_flow_arrows_on_image,put_correspondence_arrows_between_images, scipy_nn_computation, flow_from_correspondences, scipy_nn_computation_with_mask
from dense_correspondence_control.learning.testing.load_networks import load_network
import torch
import cv2
import copy



def min_max_stardardisation_to_01(image):
    image_min = np.min(image)
    image_max = np.max(image)
    range = (image_max-image_min)

    return (image-image_min)/range


def get_DoN_representation(rgb_from, rgb_to, don_from, don_to):
    rgb_from = np.round(255*rgb_from[0].detach().permute(1, 2, 0).cpu().numpy()).astype('uint8')
    rgb_to = np.round(255*rgb_to[0].detach().permute(1, 2, 0).cpu().numpy()).astype('uint8')

    don_from = don_from[0].detach().permute(1,2,0).cpu().numpy()
    don_from = np.round(255*min_max_stardardisation_to_01(don_from)).astype('uint8')
    don_to = don_to[0].detach().permute(1,2,0).cpu().numpy()
    don_to = np.round(255*min_max_stardardisation_to_01(don_to)).astype('uint8')


    final_image = np.concatenate([rgb_to, rgb_from ], axis =1)
    final_image_don = np.concatenate([don_to, don_from ], axis =1)
    final_image = np.concatenate([final_image, final_image_don], axis =0)
    return final_image



def propagate(current_rgb,bottleneck_rgb, current_seg_map, bottleneck_seg_map, networks, image_saver, writer, num_frame, use_bottleneck_mask= True) :
    # Propagate
    # Put image in torch format
    # current_rgb = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
    # bottleneck_rgb = cv2.cvtColor(bottleneck_rgb, cv2.COLOR_RGB2BGR)
    rgb_current = torch.tensor(copy.deepcopy(current_rgb)/255.).to(device).float().permute(2,0,1).unsqueeze(0)

    rgb_bottleneck = torch.tensor(copy.deepcopy(bottleneck_rgb)/255.).to(device).float().permute(2,0,1).unsqueeze(0)

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


    don_representation = get_DoN_representation(rgb_current,rgb_bottleneck,out_from, out_to)
    arrows_image = put_correspondence_arrows_between_images( np.round(255*min_max_stardardisation_to_01(out_from[0].detach().permute(1, 2, 0).cpu().numpy())).astype('uint8'),
                                                             np.round(255*min_max_stardardisation_to_01(out_to[0].detach().permute(1, 2, 0).cpu().numpy())).astype('uint8'),
                                                             correspondances, mask_from= seg_out_from.astype('uint8'), threshold=2.0,
                                             skip_amount=100)

    don_representation = np.concatenate([don_representation,arrows_image],axis =0)

    image_test = cv2.resize(image_test,(256,256) )

    don_representation = np.concatenate([don_representation,np.round(255*image_test).astype('uint8')],axis =0)


    image_saver.save_image(writer.chunk_tdir.format(chunk_id= writer.chunk_id), 'DoN_representations', num_frame, don_representation)

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
          "model_name": "cup_0100",
           # "scale_path": "./per_category_scale.json",
            "manual_scale": 0.15,
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
        "cf_set_base_color": [66./255., 206./255., 66.7/255.,1.0],
    }
},

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
            "cf_set_base_color": [0. / 255., 40. / 255., 120 / 255., 1.0],
        }
    },
]))




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
                            # "cp_manip_object": False,
                        }
                    }
                }
            },
            "mode":"once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": [ "roughness", "metallic", "specular", "anisotropic", "sheen",
                                        "clearcoat"],
            "randomisation_probabilities": [ 0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
            "metallic_min": 0.5,
            "anisotropic_min": 0.5,
            "clearcoat_min": 0.5,
            "clearcoat_roughness_min": 0.5,
            "sheen_min": 0.5,
            "keep_base_color": False,
            "relative_base_color": 0.0,
            "displacement_probability": 0.5,
            "randomisation_types": ["monochrome_random"],
            "randomisation_type_probabilities": [1.0],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },

#
# {
#         "module": "manipulators.MaterialRandomiser",
#         "config": {
#             "selector": {
#                 "provider": "getter.Material",
#                 "conditions": {
#                     "cf_use_materials_of_objects": {
#                         "provider": "getter.Entity",
#                         "conditions": {
#                             "type": "MESH",
#                             "cp_manip_object": True,
#                         }
#                     }
#                 }
#             },
#             "mode":"once_for_each",
#             "number_of_samples": 1,
#             "parameters_to_randomise": [ "base_color", "roughness", "metallic", "specular", "anisotropic", "sheen",
#                                         "clearcoat"],
#             "randomisation_probabilities": [1.0, 0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
#             "metallic_min": 0.5,
#             "anisotropic_min": 0.5,
#             "clearcoat_min": 0.5,
#             "clearcoat_roughness_min": 0.5,
#             "sheen_min": 0.5,
#             "keep_base_color": False,
#             "relative_base_color": 0.0,
#             "displacement_probability": 0.5,
#             "randomisation_types": ["image_random"],
#             "randomisation_type_probabilities": [1.0],
#             "color_textures_path": '~/Projects/resources/textures',
#             "gray_textures_path": '~/Projects/resources/gray_textures',
#         }
#     }

]))

light1, light2, _ = set_lights(
    ambient_strength_range  = [2.0, 2.0],
    light_colour_ranges = [[0., 1.0], [0.0, 0.0], [1.0, 1.0]],
    ambient_light_colour = [0.5, 0.5, 0.5, 1.0])

# DEFINE CAMERA
intrinsics = np.array([[128., 0.,63.5],
                       [0.,126.7, 63.5 ],
                       [0.,0.,1.]])
CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 128, image_height = 128,clip_start=0.001)

#DEFINE WRITER
writer = MyWriter(Utility.working_dir , 'data/test_trajectory_many_objs', has_bottleneck=True)
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
pos_volume =  ([0.,0.,0.25],[0.,0.,0.25])
orn_volume = ([0.,0.,0.],[0.,0.,0.])
bottleneck_pose, bottleneck_rgb, bottleneck_depth, bottleneck_seg_map =  process_bottleneck(num_frame = 0, chunk_id=chunk_id, pos_volume = pos_volume, orn_volume = orn_volume, manip_object = None, writer=writer)
# cv2.imwrite('./b.png', cv2.cvtColor(bottleneck_rgb, cv2.COLOR_RGB2BGR))
# bottleneck_rgb = cv2.imread('./b.png', -1)
# randomise_light_position([light1,light2], frame_num = 0) TODO: There is something weird where you can't change light location after rendering

# Define Initial camera pose
cam_pos_delta = np.array([-0.05,-0.05, 0.08])
cam_rot_delta = se3.so3_exp(np.array([0.,0.,0.9]).astype(float))
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

    current_rgb = data['colors'][0]
    current_depth = data['depth'][0]
    seg_map_cups = seg_map_cups
    bottleneck_rgb =  bottleneck_rgb
    bottleneck_depth =  bottleneck_depth
    bottleneck_seg_map = bottleneck_seg_map



    gt_cam2_image_correspondances, gt_semantics_map = writer.get_correspondance_map(intrinsics,
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
    networks, _ = load_network('/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/models/DoN/many_objects/',
        160, device=device)  # DoN_with_seg_balanced',395,


    # cv2.imwrite('./c.png', cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR))
    # current_rgb = cv2.imread('./c.png', -1)


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
                                   intrinsics=intrinsics, s_gain=0.01, xy_gain=0.01, theta_gain=0.35)


    print(twist)


    #Apply twist and set new camera pose
    camera_pose_initial_opencv = cam_from_blender_to_opencv(camera_pose)
    step = 1.0
    camera_pose_final_opencv = apply_twist(camera_pose_initial_opencv, twist, step)
    camera_pose = cam_from_blender_to_opencv(camera_pose_final_opencv)

    CameraUtility.add_camera_pose(camera_pose, num_frame)



    # Advance Frame
    #TODO: cannot randomise lights after renderging has been done.
    # randomise_light_position_colour_and_energy([light1, light2], colour_change_percent=0.5, energy_range=[50, 350],
    #                                            frame_num=num_frame)
    bpy.context.scene.frame_start = num_frame
    bpy.context.scene.frame_end = num_frame + 1

