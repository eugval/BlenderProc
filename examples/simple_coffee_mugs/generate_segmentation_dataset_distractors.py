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
from src.Eugene.dataset_utils import run_modules, fix_uv_maps,set_lights, sample_camera_pose,randomise_light_position_colour_and_energy
from src.utility.CameraUtility import CameraUtility
from src.Eugene.new_writer import MyNewWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.utility.MeshObjectUtility import MeshObject
from src.utility.filter.Filter import Filter
import matplotlib.pyplot as plt
import pickle
from src.Eugene.globals import shapenet_ids, shapenet_id_dict

temp_save_folder = '/home/eugene/test/'

Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/simple_coffee_mugs" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')

monochrome_image_percent_object = [0.2,0.8]
monochrome_image_percent_distractors = [0.3,0.7]

ambient_strength_range = [0.2,1.]
light_colour_ranges = [[0., 0.25], [0.0, 1.0], [0.8, 1.0]]
ambient_light_colour = [0.5, 0.5, 0.5, 1.0]
light_location_volume = [[-1.2, -1.2, 1.3], [1.2, 1.2, 2.3]] # 1.5
number_of_lights =  1 #np.random.choice([1,2])
print(number_of_lights)
energy_range = [450, 800]



object_name = "cup_0100"
dataset_name = 'segmentation_with_distractors'# 'test'#'segmentation_with_distractors'


# LOAD THE OBJECT INTO THE SCENE #################
# object_loading_module = Utility.initialize_modules([
#  {
#         "module": "loader.ModelNetLoader",
#         "config": {
#           "data_path": '../../resources/ModelNet40',
#           # "model_name": object_name,
#            # "scale_path": "./per_category_scale.json",
#            "manual_scale": 0.15,
#            #  'scale_range':[0.08, 0.25],
#           "train": True,
#           "samples" : 1,
#           "replacement": True,
#           "add_properties": {
#             "cp_manip_object": True,
#             "cp_object": True,
#             "cp_distractor": False,
#           },
#         #  "cf_set_shading": "SMOOTH"
#         }
#       },
# ])


object_loading_module = Utility.initialize_modules([
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
    },
])

##########################################

run_modules(object_loading_module)

object_name = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True)[0].get_name()

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
            "randomisation_type_probabilities": monochrome_image_percent_object,# [0.2,0.8][0.7,0.3],[0.5,0.5]
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))


cloth_colours_path  = '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/for_colour_extraction/segcolours.pckl'
cloth_colours = pickle.load(open(cloth_colours_path,'rb'))
cloth_colours = np.mean(np.array(cloth_colours), axis =0)/255.

cloth_colours = np.clip(cloth_colours + cloth_colours*np.random.uniform(-0.25,0.25, size =(3,)),0.,1.)

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
        "cf_set_base_color": [cloth_colours[2],cloth_colours[1],cloth_colours[0],1.0], # cloth colours is brg
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
            "randomisation_probabilities": [ 0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
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
    },]))


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

# activate depth rendering


# DO THE BOTTLENECK RENDERING
# define the pos sampling
pos_volume =  ([-0.04, -0.04, 0.15],[0.04, 0.04, 0.25]) #  ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0.,0.,0.],[0.,0.,np.pi*2])

bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1

num_frames = 2
for num_frame in range(0, num_frames):
    # Sample camera pose and set keyframe
    T_wc = sample_camera_pose(pos_volume, orn_volume, pre_selected_visible_objects = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True), fully_visible = True,obj_name_check =object_name)
    CameraUtility.add_camera_pose(T_wc, frame=num_frame)


    randomise_light_position_colour_and_energy(created_lights, colour_change_percent=0.05, energy_range =energy_range,frame_num = num_frame, location_volume=light_location_volume)
    bpy.context.scene.frame_end = num_frame + 1

RendererUtility.enable_normals_output()
RendererUtility.enable_distance_output()
RendererUtility.enable_depth_output()
RendererUtility.set_samples(50)
# ##############
# Render everything
data = RendererUtility.render(load_keys={'colors', 'depth', 'normals', 'distance'}) #'depth',
seg_data = SegMapRendererUtility.render(map_by=[ "class", "instance",  "name", "cp_manip_object"])

# Save images and camera poses
for i in range(num_frames):
    save_idx = i
    rgb = data['colors'][i]
    depth = data['depth'][i]
    distance = data['distance'][i]
    normals = data['normals'][i]

    # Save single images
    rgb_path = writer.make_path('bottleneck_rgb', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_rgb(rgb,rgb_path)

    d_path = writer.make_path('bottleneck_depth', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_depth(depth, d_path)

    distance_path = writer.make_path('bottleneck_distance', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_distance(distance, distance_path)

    n_path = writer.make_path('bottleneck_normals', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_normals(normals, n_path)


    s_path = writer.make_path('bottleneck_object_seg', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_seg_map(seg_data['instance_segmaps'][i], seg_metadata= seg_data['instance_attribute_maps'][i],  segment_by= 'manip_object', values =[1],  path = s_path)



# Delete the lights and-readd them to allow for randomistation:
for light in created_lights:
    light.delete()
###################


# LOAD DISTACTORS INTO THE SCENE
number_of_distractor = np.random.randint(0,20)
distractor_max_dist = np.random.uniform(0.1, 0.3)
distractor_offset_x_dist = np.random.uniform(-0.1,0.1)
distractor_offset_y_dist = np.random.uniform(-0.1,0.1)

object_loading_module = Utility.initialize_modules([
 # {
 #        "module": "loader.ModelNetLoader",
 #        "config": {
 #          "data_path": '../../resources/ModelNet40',
 #          # "model_name": object_name ,
 #           # "scale_path": "./per_category_scale.json",
 #           #  'category':'cup',
 #             'scale_range':[0.1,0.2],
 #          "train": True,
 #          "samples" : number_of_distractor,
 #          "replacement": True,
 #          "add_properties": {
 #            "cp_physics": True,
 #            "cp_manip_object":False,
 #            "cp_distractor":True,
 #              "cp_object": True,
 #          },
 #          "cf_set_shading": "SMOOTH"
 #        }
 #  },

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
                "max": [distractor_max_dist + distractor_offset_x_dist,distractor_max_dist+distractor_offset_y_dist, 0.],
                "min": [-distractor_max_dist + distractor_offset_x_dist, -distractor_max_dist+distractor_offset_y_dist, 0.]
            },

            "min_distance": 0.06,
            "max_distance": 1.0,
            "max_iterations": 10,
            "rot_sampler": {
                "provider": "sampler.Uniform3d",
                "max": [0, 0, 0],
                "min":[0, 0, 0], # [6.28, 6.28, 6.28]
            }
        }
    },
    # {
    #     "module": "object.PhysicsPositioning",
    #     "config": {
    #         "min_simulation_time": 4,
    #         "max_simulation_time": 8,
    #         "check_object_interval": 1
    #     }
    # },

])

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
            "randomisation_probabilities": [1.0, 0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
            "metallic_min": 0.5,
            "anisotropic_min": 0.5,
            "clearcoat_min": 0.5,
            "clearcoat_roughness_min": 0.5,
            "sheen_min": 0.5,
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


created_lights, light_colour = set_lights(
    ambient_strength_range  = ambient_strength_range,
    light_colour = light_colour,
    ambient_light_colour =ambient_light_colour,
light_location_volume = light_location_volume,
number_of_lights = number_of_lights,
energy_range = energy_range
    )


# DO THE BOTTLENECK RENDERING
# define the pos sampling
pos_volume = ([-0.14, -0.14, 0.14], [0.14, 0.14, 0.55])  #   ([-0.1, -0.1, 0.15], [0.1, 0.1, 0.45])   # ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0., 0., 0.], [0., 0., np.pi * 2])
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1

num_frames = 5
for num_frame in range(0, num_frames):
    # Sample camera pose and set keyframe
    fully_visible = np.random.choice([True, False])
    T_wc = sample_camera_pose(pos_volume, orn_volume, pre_selected_visible_objects = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True), fully_visible = fully_visible, obj_name_check =object_name)
    CameraUtility.add_camera_pose(T_wc, frame=num_frame)


    randomise_light_position_colour_and_energy(created_lights,location_volume = light_location_volume, colour_change_percent=0.05, energy_range =  energy_range,frame_num = num_frame)
    bpy.context.scene.frame_end = num_frame + 1

# RendererUtility.enable_normals_output()
# RendererUtility.enable_distance_output()
# RendererUtility.enable_depth_output()
# RendererUtility.set_samples(50)
############################
# Render everything
data = RendererUtility.render(load_keys={'colors', 'depth', 'normals', 'distance'}) #'depth',
seg_data = SegMapRendererUtility.render(map_by=["instance", "class",  "name", "cp_manip_object"])

# Save images and camera poses
for i in range(num_frames):
    save_idx = i
    rgb = data['colors'][i]
    distance = data['distance'][i]
    depth = data['depth'][i]
    normals = data['normals'][i]

    # Save single images
    rgb_path = writer.make_path('current_rgb', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_rgb(rgb,rgb_path)

    d_path = writer.make_path('current_depth', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_depth(depth, d_path)

    distance_path = writer.make_path('current_distance', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_distance(distance, distance_path)

    n_path = writer.make_path('current_normals', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_normals(normals, n_path)


    # assert manip_object_category
    s_path = writer.make_path('current_object_seg', save_idx, '.png',  chunk_id = chunk_id)

    manip_object_found = False
    for bundle in  seg_data['instance_attribute_maps'][i]:
        if(bundle['manip_object']):
            manip_object_found = True

    assert manip_object_found, 'The manip object is segmented in the image'

    writer.save_seg_map(seg_data['instance_segmaps'][i], seg_metadata= seg_data['instance_attribute_maps'][i], segment_by= 'manip_object', values =[1],   path = s_path)

    distractor_object_names = [obj.get_name()  for obj in Filter.by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects),  cp_name='distractor', value=True)]
    s_path = writer.make_path('current_distractor_seg', save_idx, '.png', chunk_id=chunk_id)
    writer.save_seg_map(seg_data['instance_segmaps'][i], seg_metadata=seg_data['instance_attribute_maps'][i],
                        segment_by='name', values=distractor_object_names, path=s_path)

    all_object_names = [obj.get_name() for obj in
                               Filter.by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects),
                                            cp_name='object', value=True)]
    s_path = writer.make_path('current_full_seg', save_idx, '.png',  chunk_id = chunk_id)
    writer.save_seg_map(seg_data['instance_segmaps'][i], seg_metadata= seg_data['instance_attribute_maps'][i], segment_by= 'name', values =all_object_names,  path = s_path)

#######################