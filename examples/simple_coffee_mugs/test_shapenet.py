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
from src.Eugene.new_writer import MySegWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.utility.MeshObjectUtility import MeshObject
from src.utility.filter.Filter import Filter
import matplotlib.pyplot as plt
import pickle
from src.utility.loader.ShapeNetLoader import ShapeNetLoader
from src.Eugene.globals import shapenet_ids,shapenet_id_dict

temp_save_folder = '/home/eugene/test/'

Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/simple_coffee_mugs" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')


shapenet_path = '../../resources/ShapeNetCore.v2'

object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ShapeNetLoader",

        "config": {
            "synset_id":"02843684",
            "used_source_id":"2e81196fa81cc1fba29538f582b2005",
            # "synset_ids":shapenet_ids,
          "data_path": shapenet_path,
           "manual_scale": 0.15,
           #  'scale_range':[0.08, 0.25],
          "samples" : 1,
          "replacement": True,
          "add_properties": {
            "cp_manip_object": True,
            "cp_object": True,
            "cp_distractor": False,
          },
        }
      },
])




run_modules(object_loading_module)

object_name = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True)[0].get_name()
#


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
            "randomisation_type_probabilities": [0.2,0.8],# [0.2,0.8][0.7,0.3],[0.5,0.5]
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))












# LOAD DISTACTORS INTO THE SCENE
number_of_distractor = np.random.randint(0, 15)
distractor_max_dist = np.random.uniform(0.1, 0.3)
distractor_offset_x_dist = np.random.uniform(-0.1,0.1)
distractor_offset_y_dist = np.random.uniform(-0.1,0.1)

object_loading_module = Utility.initialize_modules([

{
        "module": "loader.ShapeNetLoader",

        "config": {
            "synset_ids":shapenet_ids,
          "data_path": shapenet_path,
           "manual_scale": 0.15,
           #  'scale_range':[0.08, 0.25],
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

            "min_distance": 0.02,
            "max_distance": 1.0,
            "max_iterations": 10,
            "rot_sampler": {
                "provider": "sampler.Uniform3d",
                "max": [0, 0, 0],
                "min": [0, 0, 0]
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
            "reference_texture_prob": 0.35,
            "store_reference_texture": False,
            "randomisation_types": [ "monochrome_random", "image_random"], #  "image_random"
            "randomisation_type_probabilities": [0.35, 0.65], #
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))
























print('done')

# object_name = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True)[0].get_name()