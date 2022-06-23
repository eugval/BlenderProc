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
dataset_name = 'segmentation_dataset_simple_textures_10k'
# LOAD THE OBJECT INTO THE SCENE
object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ModelNetLoader",
        "config": {
          "data_path": '../../resources/ModelNet40',
          #"model_name": object_name ,
           # "scale_path": "./per_category_scale.json",
           "manual_scale": 0.15,
            # 'scale_range':[0.15,0.2],
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
        "cf_set_base_color": [0./255., 40./255., 120/255.,1.0],
    }
},
]))


# Randoimse the walls
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
            "keep_base_color": False,
            # "relative_base_color": 0.2,
            "displacement_probability": 0.5,
            "randomisation_types": ["monochrome_random"],
            "randomisation_type_probabilities": [1.0],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    },]))

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
            "randomisation_types": ["monochrome_random"], #  "image_random"
            "randomisation_type_probabilities": [1.0], # [0.2,0.8]
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
# RendererUtility.enable_normals_output()


# DEFINE SAMPLING REGION
pos_volume =  ([-0.04, -0.04, 0.15],[0.04, 0.04, 0.25]) #  ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
orn_volume = ([0.,0.,0.],[0.,0.,np.pi*2])


camera_poses = []


bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = 1

num_frames = 10
for num_frame in range(0, num_frames):

    if(num_frame>4):
        # Increase sampling volume for rest of thingy
        pos_volume = ([-0.07, -0.07, 0.15],
                      [0.07, 0.07, 0.75])  # ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
        orn_volume = ([0., 0., 0.], [0., 0.,np.pi*2])

    #Sample camera pose and set keyframe
    T_wc = sample_camera_pose(pos_volume, orn_volume, pre_selected_visible_objects = Filter.by_cp(elements= MeshObject.convert_to_meshes(bpy.data.objects) ,cp_name='manip_object',value=True), fully_visible =False)
    CameraUtility.add_camera_pose(T_wc, frame=num_frame)
    camera_pose_in_world = np.array(WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix'))
    assert np.isclose(camera_pose_in_world, T_wc).all()
    camera_poses.append(T_wc)

    randomise_light_position_colour_and_energy([light1,light2], colour_change_percent=0.3, energy_range = [50,350],frame_num = num_frame)

    bpy.context.scene.frame_end = num_frame + 1


# Render everything
data = RendererUtility.render(load_keys={'colors', 'depth', 'normals'})
seg_data = SegMapRendererUtility.render(map_by=["instance", "class"])


# Save intrinsics
writer.save_camera_intrinsics(intrinsics, chunk_id)

#TODO: Need to have segementation with just 0-1

# Save images and camera poses
for i in range(5):
    save_idx = i
    rgb = data['colors'][i]
    depth = data['depth'][i]
    # normals = data['normals'][i]

    # Save single images
    rgb_path = writer.make_path(save_idx, '.png', 'bottleneck_rgb')
    writer.save_rgb(rgb, i, chunk_id, path=rgb_path)

    d_path = writer.make_path(save_idx, '.png', 'bottleneck_depth')
    writer.save_depth(depth, i, chunk_id, path=d_path)

    # n_path = writer.make_path(save_idx, '.png', 'bottleneck_normal')
    # writer.save_normals(normals, i, chunk_id, path=n_path)

    s_path = writer.make_path(save_idx, '.png', 'bottleneck_segmentation')
    writer.save_seg_map(seg_data['instance_segmaps'][i], seg_data['instance_attribute_maps'][i], i, chunk_id,
                        path=s_path)

    writer.save_camera_pose(camera_poses[i], i, chunk_id)



# Save images and camera poses
for i in range(5,10):
    save_idx = i%5
    rgb = data['colors'][i]
    depth = data['depth'][i]
    # normals = data['normals'][i]

    # Save single images
    rgb_path = writer.make_path(save_idx,'.png','current_rgb')
    writer.save_rgb(rgb,i,chunk_id, path = rgb_path)

    d_path = writer.make_path(save_idx, '.png', 'current_depth')
    writer.save_depth(depth,i,chunk_id, path = d_path)

    # n_path = writer.make_path(save_idx, '.png', 'current_normal')
    # writer.save_normals(normals, i, chunk_id, path = n_path)

    s_path = writer.make_path(save_idx, '.png', 'current_segmentation')
    writer.save_seg_map(seg_data['instance_segmaps'][i],seg_data['instance_attribute_maps'][i], i, chunk_id, path = s_path)

    writer.save_camera_pose(camera_poses[i],i,chunk_id)

# Compute and save correspondances
for i in range(5):
    for j in range(5,num_frames):
        writer.get_and_save_correspondances(i,j, data['depth'],camera_poses,chunk_id,intrinsics)


