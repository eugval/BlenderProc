from src.utility.SetupUtility import SetupUtility
argv = SetupUtility.setup([   ]) #'pydevd-pycharm~=193.6911.2'
from src.utility.LockerUtility import Locker

import os
import bpy
import sys
import numpy as np
from src.utility.Utility import Utility
from src.utility.Initializer import Initializer
from src.utility.loader.BlendLoader import BlendLoader
from src.Eugene.dataset_utils import run_modules, fix_uv_maps,set_lights, sample_camera_pose,randomise_light_position_colour_and_energy, cam_from_blender_to_opencv, project_object_top_point, sample_camera_pose_delta, light_setting_selection
from src.utility.CameraUtility import CameraUtility
from src.Eugene.new_writer import MyNewWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.utility.MeshObjectUtility import MeshObject
from src.utility.filter.Filter import Filter
import matplotlib.pyplot as plt
import pickle
from src.Eugene.globals import shapenet_ids, shapenet_id_dict, shapenet_class_name_dict, class_to_onehot_shapenet, onehot_to_class_shapenet, class_to_onehot_combined
from src.Eugene import se3_tools as se3
import copy
from PIL import Image

def resize_with_PIL(pil_image, length):
    dim0 = pil_image.size[0]
    dim1 = pil_image.size[1]
    r = dim1 / dim0
    im = pil_image.resize((length, int(length * r)))
    return im



dataset_name = 'test'
temp_save_folder = '/home/eugene/test/'

Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/simple_coffee_mugs" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')

# monochrome percent, image random percent
monochrome_image_percent_object = [0.15,0.85] #[0.2,0.8]
monochrome_image_percent_distractors = [0.2,0.8] # [0.3,0.7]

object_model_dataset = np.random.choice(['modelnet','shapenet'], p =[0.5,0.5])

light_colour_ranges = [[0., 0.25], [0.0, 1.0], [0.8, 1.0]]
ambient_light_colour = [0.5, 0.5, 0.5, 1.0]
# ambient_strength_range = [0.2, 2.0]
# light_location_volume = [[-1.5, -1.5, 0.3], [1.5, 1.5 , 2.4]]
# number_of_lights =  np.random.choice([1,2])
# print(number_of_lights)
# energy_range = [50, 800]

lights_enabled = False
light_setting = ['mostly_ambient']# ['mostly_ambient', 'strong_top_shadow', 'generic_shadow', 'very_bright', 'very_dim']

light_setting_choice =  np.random.choice(light_setting, p = [0.1,0.4, 0.4, 0.05,0.05])

ambient_strength_range,light_location_volume,number_of_lights,energy_range = light_setting_selection(light_setting_choice)

object_name = None#"cup_0100"
# 'test'#'segmentation_with_distractors'
add_bottleneck_pose_in_current = True # Makes the first of the current frames being the same as the bottleneck frame
relative_to_top = False # THis is used for the bottleneck camera positioning
successful_completion = True

# LOAD THE OBJECT INTO THE SCENE #################

loader =  []

if object_model_dataset == 'modelnet':

    loader.append(
        {
            "module": "loader.ModelNetLoader",
            "config": {
              "data_path": '../../resources/ModelNet40',
              # "model_name": object_name,
               # "scale_path": "./per_category_scale.json",
               "manual_scale": 0.15,
               #  'scale_range':[0.08, 0.25],
              "train": True,
              "samples" : 1,
              "replacement": True,
              "add_properties": {
                "cp_manip_object": True,
                "cp_object": True,
                "cp_distractor": False,
              },
            #  "cf_set_shading": "SMOOTH"
            }
          }
    )
elif(object_model_dataset =='shapenet'):
    loader.append(
        {
        "module": "loader.ShapeNetLoader",

        "config": {
            "synset_ids":shapenet_ids ,
            # "used_source_id": "2e81196fa81cc1fba29538f582b2005",
            "data_path": '../../resources/ShapeNetCore.v2',
            "manual_scale": 0.15,
             # 'scale_range':[0.1, 0.2],
            "samples": 1,
            "replacement": True,
            "add_properties": {
                "cp_manip_object": True,
                "cp_object": True,
                "cp_distractor": False,
            },
        }
    }
    )


object_loading_module = Utility.initialize_modules(loader)

run_modules(object_loading_module)

object_name = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True)[0].get_name()
object_class = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True)[0].get_cp('category_id')

object_class_id = shapenet_class_name_dict[object_class] if (object_model_dataset =='shapenet') else 0
object_class_onehot = class_to_onehot_combined[object_class]


object_info_dict = {
    'name' : object_name,
    'class': object_class,
    'class_onehot': object_class_onehot,
    'class_id': object_class_id
}

print(object_info_dict)



# # DEFINE VISUALS
fix_uv_maps()

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
            "randomisation_probabilities": [1.0, 0.5, 0.3, 0.3, 0.3, 0.15, 0.05],
            "metallic_min": 0.6,
            "anisotropic_min": 0.6,
            "clearcoat_min": 0.6,
            "clearcoat_roughness_min": 0.6,
            "sheen_min": 0.6,
            "keep_base_color": True,
            # "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "reference_texture_prob": 0.,
            "store_reference_texture": True,
            "randomisation_types": ["monochrome_random",  "image_random"], #  "image_random"
            "randomisation_type_probabilities": monochrome_image_percent_object,# [0.2,0.8][0.7,0.3],[0.5,0.5]
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))


cloth_colours_path  = '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/for_colour_extraction/segcolours.pckl'
cloth_colours = pickle.load(open(cloth_colours_path,'rb'))
cloth_colours = np.array(cloth_colours)
# BGR-> RGB convertion
# tmp = copy.deepcopy(cloth_colours[:,2])
# cloth_colours[:,2] = cloth_colours[:,0]
# cloth_colours[:,0]= tmp

cloth_colours = np.mean(cloth_colours, axis =0)/255.

cloth_colours = np.clip(cloth_colours + cloth_colours*np.random.uniform(-0.25,0.25, size =(3,)),0.,1.)
# cloth_colours=np.array([1.0,0.,0.])

# Randoimse the walls
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
        "cf_set_base_color": [cloth_colours[2],cloth_colours[1],cloth_colours[0],1.0], # cloth colours is bgr
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
            "mode":"once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": [   "roughness", "metallic", "specular", "anisotropic", "sheen",
                                        "clearcoat"],
            "randomisation_probabilities": [ 0.5, 0.3, 0.3, 0.3, 0.15, 0.05],
            "metallic_min": 0.6,
            "anisotropic_min": 0.6,
            "clearcoat_min": 0.6,
            "clearcoat_roughness_min": 0.6,
            "sheen_min": 0.6,
            "keep_base_color": True,
            # "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "randomisation_types": ["monochrome_random"],
            "randomisation_type_probabilities": [1.0],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))


if(lights_enabled):
    created_lights, light_colour = set_lights(
        ambient_strength_range  = ambient_strength_range,
        light_colour_ranges = light_colour_ranges,
        ambient_light_colour = ambient_light_colour,
        light_location_volume=light_location_volume,
        number_of_lights=number_of_lights,
        energy_range =energy_range)



intrinsics = np.array([[128., 0.,63.5],
                       [0.,126.7, 63.5 ],
                       [0.,0.,1.]])

CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 128, image_height = 128,clip_start=0.001)

#DEFINE WRITER
writer = MyNewWriter(Utility.working_dir , '../data/datasets/{}'.format(dataset_name))
chunk_id = writer.check_and_create_trajectory_folders()
writer.save_object_info(object_info_dict, os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'object_info.pckl'))
writer.save_object_info({'light_setting': light_setting_choice}, os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'light_setting.pckl'))
writer.save_camera_intrinsics(intrinsics,os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'intrinsics.npy'))



# DO THE BOTTLENECK RENDERING
# define the pos sampling
pos_volume =  ([-0.05, -0.05, 0.12],[0.05, 0.05, 0.25])   #([-0.05, -0.05, 0.12],[0.05, 0.05, 0.25])   #  ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0.,0.,-np.pi],[0.,0.,np.pi])

bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1

num_bottleneck_frames = 1 #TODO: WARNING: THIS CANNOT BE CHANGED OR THE POSES IN CURRENT ARE WRONG
bottleneck_camera_poses = []
bottleneck_object_pixel_positions = []
for num_frame in range(0, num_bottleneck_frames):
    # Sample camera pose and set keyframe
    relative_to_top = relative_to_top
    T_wc = sample_camera_pose(pos_volume, orn_volume,
                              pre_selected_visible_objects = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True),
                              fully_visible = True ,obj_name_check = object_name, relative_to_top=relative_to_top)

    CameraUtility.add_camera_pose(T_wc, frame=num_frame)

    if(lights_enabled):
        randomise_light_position_colour_and_energy(created_lights, colour_change_percent=0.05, energy_range =energy_range,frame_num = num_frame, location_volume=light_location_volume)
    object_xy_pixel_position = project_object_top_point(Filter.one_by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects), cp_name='manip_object',value=True), T_wc)
    bottleneck_object_pixel_positions.append(object_xy_pixel_position)
    bottleneck_camera_poses.append(cam_from_blender_to_opencv(T_wc))

    bpy.context.scene.frame_end = num_frame + 1

RendererUtility.enable_normals_output()
RendererUtility.enable_distance_output()
RendererUtility.enable_depth_output()
RendererUtility.set_samples(50)
# ##############
# Render everything
bottleneck_data = RendererUtility.render(load_keys={'colors', 'depth'}) # 'depth', 'normals', 'distance'
bottleneck_seg_data = SegMapRendererUtility.render(map_by=[ "class", "instance",  "name", "cp_manip_object"])



# Save images and camera poses
for i in range(num_bottleneck_frames):
    save_idx = i
    rgb = bottleneck_data['colors'][i]
    depth = bottleneck_data['depth'][i]
    # distance = bottleneck_data['distance'][i]
    # normals = bottleneck_data['normals'][i]

    # Save single images
    rgb_path = writer.make_path('bottleneck_rgb', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_rgb(rgb,rgb_path)

    rgb_64 = np.asarray(resize_with_PIL(Image.fromarray(rgb),64))
    rgb_path_64 = writer.make_path('bottleneck_rgb_64', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_rgb(rgb_64,rgb_path_64)

    d_path = writer.make_path('bottleneck_depth', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_depth(depth, d_path)
    #
    # distance_path = writer.make_path('bottleneck_distance', save_idx, '.png',  chunk_id = chunk_id)
    # writer.save_distance(distance, distance_path)
    #
    # n_path = writer.make_path('bottleneck_normals', save_idx, '.png',  chunk_id = chunk_id)
    # writer.save_normals(normals, n_path)

    s_path = writer.make_path('bottleneck_object_seg', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_seg_map(bottleneck_seg_data['instance_segmaps'][i], seg_metadata= bottleneck_seg_data['instance_attribute_maps'][i],  segment_by= 'manip_object', values =[1],  path = s_path)


    s_path_64 = writer.make_path('bottleneck_object_seg_64', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_seg_map(bottleneck_seg_data['instance_segmaps'][i],
                        seg_metadata= bottleneck_seg_data['instance_attribute_maps'][i],
                        segment_by= 'manip_object', values =[1],  path = s_path_64, resize_size=64)



    bottleneck_pose_path =  os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'bottleneck_camera_poses.pckl')
    writer.save_camera_pose(bottleneck_camera_poses[i],bottleneck_pose_path,'{:06d}'.format(save_idx))

    bottleneck_object_pixel_positions_path =  os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'bottleneck_pixel_position.pckl')
    writer.save_camera_pose(bottleneck_object_pixel_positions[i],bottleneck_object_pixel_positions_path,'{:06d}'.format(save_idx))




### Save bottleneck-bottleneck_correspondences
# Compute and save correspondances
for i in range(num_bottleneck_frames):
    for j in range(i+1,num_bottleneck_frames):
        correspondence_path = writer.make_path('bottleneck_to_bottleneck_correspondences',  chunk_id = chunk_id, file_name= '{:06d}_{:06d}.png'.format(i,j))
        semantic_map_path = writer.make_path('bottleneck_to_bottleneck_semantic_map',  chunk_id = chunk_id, file_name= '{:06d}_{:06d}.png'.format(i,j))
        correspondence_map, semantic_map = writer.get_and_save_correspondances(i,j, bottleneck_data['depth'],bottleneck_camera_poses,intrinsics, correspondence_path, semantic_map_path)




# Delete the lights and-readd them to allow for randomistation:
if(lights_enabled):
    for light in created_lights:
        light.delete()
###################


# LOAD DISTACTORS INTO THE SCENE
number_of_distractor = np.random.randint(0,25)
distractor_max_dist = np.random.uniform(0.1, 0.3)
distractor_offset_x_dist = np.random.uniform(-0.1,0.1)
distractor_offset_y_dist = np.random.uniform(-0.1,0.1)




distractor_loader = []

if(object_model_dataset =='modelnet'):
    distractor_loader.append(
        {
            "module": "loader.ModelNetLoader",
            "config": {
                "data_path": '../../resources/ModelNet40',
                # "model_name": object_name ,
                # "scale_path": "./per_category_scale.json",
                #  'category':'cup',
                'scale_range': [0.1, 0.2],
                "train": True,
                "samples": number_of_distractor,
                "replacement": True,
                "add_properties": {
                    "cp_physics": True,
                    "cp_manip_object": False,
                    "cp_distractor": True,
                    "cp_object": True,
                },
                "cf_set_shading": "SMOOTH"
            }
        },
    )
elif(object_model_dataset =='shapenet'):
    distractor_loader.append(
        {
                "module": "loader.ShapeNetLoader",

                "config": {
                    "synset_ids":shapenet_ids,
                  "data_path": '../../resources/ShapeNetCore.v2',
                   "manual_scale": 0.15,
                   # 'scale_range':[0.1, 0.2],
                  "samples" : number_of_distractor,
                  "replacement": True,
                  "add_properties": {
                    "cp_physics": True,
                    "cp_manip_object":False,
                    "cp_distractor":True,
                      "cp_object": True,
                  },
                }
              },
    )
else:
    raise NotImplementedError()


distractor_loader.append(
    {
        "module": "object.OnSurfaceSampler",
        "config": {
            "objects_to_sample": {
                "provider": "getter.Entity",
                "conditions": {
                    "cp_distractor": True,
                }
            },
            "surface": {
                "provider": "getter.Entity",
                "index": 0,
                "conditions": {
                    "name": "ground_plane0"
                }
            },
            "pos_sampler": {
                "provider": "sampler.Uniform3d",
                "max": [distractor_max_dist + distractor_offset_x_dist, distractor_max_dist + distractor_offset_y_dist,
                        0.],
                "min": [-distractor_max_dist + distractor_offset_x_dist,
                        -distractor_max_dist + distractor_offset_y_dist, 0.]
            },

            "min_distance": 0.06,
            "max_distance": 1.0,
            "max_iterations": 10,
            "rot_sampler": {
                "provider": "sampler.Uniform3d",
                "max": [0, 0, 0],
                "min": [0, 0, 0],  # [6.28, 6.28, 6.28]
            }
        }
    }
)

object_loading_module = Utility.initialize_modules(distractor_loader)

run_modules(object_loading_module)



# # DEFINE VISUALS
fix_uv_maps()


# Randomise the distractors
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
                            "cp_distractor": True,
                        }
                    }
                }
            },
            "mode":"once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": [ "base_color",  "roughness", "metallic", "specular", "anisotropic", "sheen",
                                        "clearcoat"],
            "randomisation_probabilities": [1.0, 0.5, 0.3, 0.3, 0.3, 0.15, 0.05],
            "metallic_min": 0.6,
            "anisotropic_min": 0.6,
            "clearcoat_min": 0.6,
            "clearcoat_roughness_min": 0.6,
            "sheen_min": 0.6,
            "keep_base_color": True,
            # "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "reference_texture_prob": 0.25,
            "store_reference_texture": False,
            "randomisation_types": [ "monochrome_random", "image_random"], #  "image_random"
            "randomisation_type_probabilities": monochrome_image_percent_distractors, #  [0.35,0.65],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))



if(lights_enabled):
    change_light_setting = np.random.choice([True,False], p =[0.25,0.75])
    if(change_light_setting):
        light_setting.remove(light_setting_choice)

        light_setting_choice = np.random.choice(light_setting)
        ambient_strength_range,light_location_volume,number_of_lights,energy_range = light_setting_selection(light_setting_choice)




    created_lights, light_colour = set_lights(
        ambient_strength_range  = ambient_strength_range,
        light_colour = light_colour,
        ambient_light_colour = ambient_light_colour,
    light_location_volume = light_location_volume,
    number_of_lights = number_of_lights,
    energy_range = energy_range
        )


# DO THE BOTTLENECK RENDERING
# define the pos sampling

bottleneck_camera_pose = cam_from_blender_to_opencv(bottleneck_camera_poses[0])



pos_volume = ([-0.11, -0.11, -0.06], [0.11, 0.11, 0.26])  # ([-0.1, -0.1, -0.06], [0.1, 0.1, 0.25])  ([-0.1, -0.1, 0.15], [0.1, 0.1, 0.45])   # ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0., 0., -np.pi/2], [0., 0., np.pi/2])
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1

current_num_frames = 5
current_camera_poses = []
current_object_pixel_positions = []
for num_frame in range(0, current_num_frames):
    # Sample camera pose and set keyframe
    relative_to_top = relative_to_top
    fully_visible = np.random.choice([True, False])

    if(num_frame==0 and add_bottleneck_pose_in_current):
        T_wc = copy.deepcopy(bottleneck_camera_pose)
    else:
        T_wc = sample_camera_pose_delta(copy.deepcopy(bottleneck_camera_pose), pos_volume, orn_volume,
                                  pre_selected_visible_objects = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,
                                                                              cp_name='manip_object',value=True),
                                  fully_visible = fully_visible,
                                  obj_name_check =object_name,
                                 )

    CameraUtility.add_camera_pose(T_wc, frame=num_frame)

    if(lights_enabled):
        randomise_light_position_colour_and_energy(created_lights,location_volume = light_location_volume, colour_change_percent=0.05, energy_range =  energy_range,frame_num = num_frame)

    object_xy_pixel_position = project_object_top_point(
        Filter.one_by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects), cp_name='manip_object', value=True), T_wc)

    current_object_pixel_positions.append(object_xy_pixel_position)
    current_camera_poses.append(cam_from_blender_to_opencv(T_wc))

    bpy.context.scene.frame_end = num_frame + 1

# RendererUtility.enable_normals_output()
# RendererUtility.enable_distance_output()
# RendererUtility.enable_depth_output()
# RendererUtility.set_samples(50)
############################
# Render everything
current_data = RendererUtility.render(load_keys={'colors', 'depth',}) #'depth',  'normals', 'distance'
current_seg_data = SegMapRendererUtility.render(map_by=["instance", "class",  "name", "cp_manip_object"])


#TODO: ADD THE RESIZING
# Save images and camera poses
for i in range(current_num_frames):
    save_idx = i
    rgb = current_data['colors'][i]
    # distance = current_data['distance'][i]
    depth = current_data['depth'][i]
    # normals = current_data['normals'][i]

    # # correct for depth artifacts
    # depth[depth > 1.] = 1.0

    # Save single images
    rgb_path = writer.make_path('current_rgb', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_rgb(rgb,rgb_path)

    # Save single images with 64x64
    rgb_64 = np.asarray(resize_with_PIL(Image.fromarray(rgb),64))
    rgb_path_64 = writer.make_path('current_rgb_64', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_rgb(rgb_64,rgb_path_64)

    d_path = writer.make_path('current_depth', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_depth(depth, d_path)

    # distance_path = writer.make_path('current_distance', save_idx, '.png',  chunk_id = chunk_id)
    # writer.save_distance(distance, distance_path)
    #
    # n_path = writer.make_path('current_normals', save_idx, '.png',  chunk_id = chunk_id)
    # writer.save_normals(normals, n_path)


    # assert manip_object_category
    s_path = writer.make_path('current_object_seg', save_idx, '.png',  chunk_id = chunk_id)

    manip_object_found = False
    for bundle in  current_seg_data['instance_attribute_maps'][i]:
        if(bundle['manip_object']):
            manip_object_found = True
            break

    assert manip_object_found, 'The manip object is segmented in the image'

    writer.save_seg_map(current_seg_data['instance_segmaps'][i], seg_metadata= current_seg_data['instance_attribute_maps'][i], segment_by= 'manip_object', values =[1],   path = s_path)



    ## 64x64 segmentation
    s_path_64 = writer.make_path('current_object_seg_64', save_idx, '.png',  chunk_id = chunk_id)

    writer.save_seg_map(current_seg_data['instance_segmaps'][i],
                        seg_metadata= current_seg_data['instance_attribute_maps'][i],
                        segment_by= 'manip_object', values =[1],   path = s_path_64, resize_size=64)



    distractor_object_names = [obj.get_name()  for obj in Filter.by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects),  cp_name='distractor', value=True)]
    s_path = writer.make_path('current_distractor_seg', save_idx, '.png', chunk_id=chunk_id)
    writer.save_seg_map(current_seg_data['instance_segmaps'][i], seg_metadata=current_seg_data['instance_attribute_maps'][i],
                        segment_by='name', values=distractor_object_names, path=s_path, check_degenerate=False)


    ### 64x64 distractor segmentation
    s_path_64 = writer.make_path('current_distractor_seg_64', save_idx, '.png', chunk_id=chunk_id)
    writer.save_seg_map(current_seg_data['instance_segmaps'][i],
                        seg_metadata=current_seg_data['instance_attribute_maps'][i],
                        segment_by='name', values=distractor_object_names, path=s_path_64, resize_size=64,check_degenerate= False)


    # ### 64x64 single distractor segmentation
    # s_path_64 = writer.make_path('single_distractor_seg_64', save_idx, '.png', chunk_id=chunk_id)
    # single_distractor = [ Filter.one_by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects),  cp_name='distractor', value=True).get_name()] TODO: this can select distractors out of image
    # writer.save_seg_map(current_seg_data['instance_segmaps'][i],
    #                     seg_metadata=current_seg_data['instance_attribute_maps'][i],
    #                     segment_by='name', values=np.random.choice(distractor_object_names), path=s_path_64, resize_size=64,check_degenerate= False)



    all_object_names = [obj.get_name() for obj in
                               Filter.by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects),
                                            cp_name='object', value=True)]
    s_path = writer.make_path('current_full_seg', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_seg_map(current_seg_data['instance_segmaps'][i], seg_metadata= current_seg_data['instance_attribute_maps'][i], segment_by= 'name', values =all_object_names,  path = s_path)


    s_path_64 = writer.make_path('current_full_seg_64', save_idx, '.png', chunk_id=chunk_id)
    writer.save_seg_map(current_seg_data['instance_segmaps'][i],
                        seg_metadata=current_seg_data['instance_attribute_maps'][i], segment_by='name',
                        values=all_object_names, path=s_path_64, resize_size=64, check_degenerate= False)


    s_instance_path= writer.make_path('current_instance_seg', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_full_seg_map(current_seg_data['instance_segmaps'][i],
                        seg_metadata=current_seg_data['instance_attribute_maps'][i], path=s_instance_path)


    s_instance_path_64= writer.make_path('current_instance_seg_64', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_full_seg_map(current_seg_data['instance_segmaps'][i],
                        seg_metadata=current_seg_data['instance_attribute_maps'][i], path=s_instance_path_64, resize_size=64)


    current_pose_path =  os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'current_camera_poses.pckl')
    writer.save_camera_pose(current_camera_poses[i],current_pose_path,'{:06d}'.format(save_idx))

    current_object_pixel_positions_path =  os.path.join(writer.chunk_tdir.format(chunk_id = chunk_id), 'current_pixel_position.pckl')
    writer.save_camera_pose(current_object_pixel_positions[i],current_object_pixel_positions_path,'{:06d}'.format(save_idx))





#######################




### Save current-current_correspondences and pose deltas
# Compute and save correspondances
for i in range(current_num_frames):
    for j in range(i+1,current_num_frames):
        correspondence_path = writer.make_path('current_to_current_correspondences',  chunk_id = chunk_id, file_name= '{:06d}_{:06d}.png'.format(i,j))
        semantic_map_path = writer.make_path('current_to_current_semantic_map',  chunk_id = chunk_id, file_name= '{:06d}_{:06d}.png'.format(i,j))
        correspondence_map, semantic_map =  writer.get_and_save_correspondances(i,j, current_data['depth'],current_camera_poses,intrinsics, correspondence_path, semantic_map_path)





### Save current-bottleneck_correspondences
# Compute and save correspondances
for i in range(current_num_frames):
    for j in range(num_bottleneck_frames):
        correspondence_path = writer.make_path('current_to_bottleneck_correspondences',  chunk_id = chunk_id, file_name= '{:06d}_{:06d}.png'.format(i,j))
        semantic_map_path = writer.make_path('current_to_bottleneck_semantic_map',  chunk_id = chunk_id, file_name= '{:06d}_{:06d}.png'.format(i,j))
        correspondence_map, semantic_map = writer.get_and_save_correspondances(i,j, current_data['depth'],current_camera_poses,intrinsics,
                                                                               correspondence_path, semantic_map_path,
                                                                               depth_for_j=bottleneck_data['depth'], camera_poses_for_j=bottleneck_camera_poses
                                                                               )

        current_to_bottleneck_pose_delta = se3.pose_inv(current_camera_poses[i]).dot(bottleneck_camera_poses[j])
        current_to_bottleneck_pose_delta_path = os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'bottleneck_poses_in_current_frame.pckl')
        writer.save_camera_pose(current_to_bottleneck_pose_delta, current_to_bottleneck_pose_delta_path, '{:06d}_{:06d}'.format(i,j))

        current_to_bottleneck_z = se3.rot2euler('XYZ', current_to_bottleneck_pose_delta[:3, :3])
        print(current_to_bottleneck_z)
        current_to_bottleneck_z_path =  os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'bottleneck_z_angles_in_current_frame.pckl')
        writer.save_camera_pose(current_to_bottleneck_z, current_to_bottleneck_z_path, '{:06d}_{:06d}'.format(i,j))

        current_to_bottleneck_z_path =  os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'bottleneck_cos_sin_angles_in_current_frame.pckl')
        writer.save_camera_pose(np.array([np.sin(current_to_bottleneck_z[2]), np.cos(current_to_bottleneck_z[2])]), current_to_bottleneck_z_path, '{:06d}_{:06d}'.format(i,j))



        current_to_bottleneck_pixel_position = bottleneck_object_pixel_positions[j] - current_object_pixel_positions[i]
        current_to_bottleneck_pixel_position_path =  os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'current_to_bottleneck_pixel_pos_delta.pckl')
        writer.save_camera_pose(current_to_bottleneck_pixel_position, current_to_bottleneck_pixel_position_path, '{:06d}_{:06d}'.format(i,j))



np.save( os.path.join(writer.chunk_tdir.format(chunk_id=chunk_id), 'successful_completion.npy'), successful_completion)