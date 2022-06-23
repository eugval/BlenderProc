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
from src.Eugene.my_writer import MyWriter
from src.utility.RendererUtility import RendererUtility
from src.utility.WriterUtility import WriterUtility#
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.utility.MeshObjectUtility import MeshObject
from src.utility.filter.Filter import Filter

temp_save_folder = '/home/eugene/test/'

Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/simple_coffee_mugs" + "/"
Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('../assets/scenes/room.blend')

object_name = "cup_0100"
dataset_name = 'cups_randomised'
# LOAD THE OBJECT INTO THE SCENE
object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ModelNetLoader",
        "config": {
          "data_path": '../../resources/ModelNet40',
          "model_name": object_name ,
           # "scale_path": "./per_category_scale.json",
            "manual_scale": 0.15,

          "train": True,
          "samples" : 1,
          "replacement": True,
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
        "cf_set_base_color": [0./255., 40./255., 120/255.,1.0],
    }
},
]))


#
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
            "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "randomisation_types": ["monochrome_random"],
            "randomisation_type_probabilities": [1.0],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))



light1, light2 = set_lights(
    ambient_strength_range  = [2.0, 2.0],
    light_colour_ranges = [[0., 1.0], [0.0, 0.0], [1.0, 1.0]],
    ambient_light_colour = [0.5, 0.5, 0.5, 1.0])

# DEFINE CAMERA
# intrinsics = np.array([[355.5556369357639, 0., 127.5],
#                      [0,  355.5556369357639, 127.5],
#                      [0., 0., 1.]])

intrinsics = np.array([[128., 0.,63.5],
                       [0.,126.7, 63.5 ],
                       [0.,0.,1.]])

CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 128, image_height = 128,clip_start=0.001)



#DEFINE WRITER
writer = MyWriter(Utility.working_dir , '../data/datasets/{}'.format(dataset_name))
chunk_id = writer.check_and_create_trajectory_folders()
writer.save_camera_intrinsics(CameraUtility.get_intrinsics_as_K_matrix(), chunk_id=chunk_id)

# activate depth rendering
RendererUtility.set_samples(50)
RendererUtility.enable_depth_output()


# DEFINE SAMPLING REGION
pos_volume =  ([-0.07,-0.07,0.15],[0.07,0.07, 0.55]) #  ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0.,0.,0.],[0.,0.,6.283185307])


camera_poses = []


bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1

num_frames = 10
for num_frame in range(0, num_frames):
    #Sample camera pose and set keyframe
    T_wc = sample_camera_pose(pos_volume, orn_volume, objects_fully_visible = [object_name])
    CameraUtility.add_camera_pose(T_wc, frame=num_frame)
    camera_pose_in_world = np.array(WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix'))
    assert np.isclose(camera_pose_in_world, T_wc).all()
    camera_poses.append(T_wc)

    randomise_light_position_colour_and_energy([light1,light2], colour_change_percent=0.5, energy_range = [50,350],frame_num = num_frame)

    bpy.context.scene.frame_end = num_frame + 1


# Render everything
data = RendererUtility.render(load_keys={'colors', 'depth'})
seg_data = SegMapRendererUtility.render(map_by=["instance", "class"])


# Save intrinsics
writer.save_camera_intrinsics(intrinsics, chunk_id)

# Save images and camera poses
for i in range(num_frames):
    rgb = data['colors'][i]
    depth = data['depth'][i]

    # Save single images
    writer.save_rgb(rgb,i,chunk_id)
    writer.save_depth(depth,i,chunk_id)
    writer.save_camera_pose(camera_poses[i],i,chunk_id)
    writer.save_seg_map(seg_data['instance_segmaps'][i],seg_data['instance_attribute_maps'][i], i, chunk_id)


# Compute and save correspondances
for i in range(num_frames):
    for j in range(i+1,num_frames):
        writer.get_and_save_correspondances(i,j, data['depth'],camera_poses,chunk_id,intrinsics)


