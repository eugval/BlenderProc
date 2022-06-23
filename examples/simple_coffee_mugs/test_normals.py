from src.utility.SetupUtility import SetupUtility
SetupUtility.setup([])

import argparse
from src.utility.loader.BlendLoader import BlendLoader
from src.utility.WriterUtility import WriterUtility
from src.utility.Initializer import Initializer
from src.utility.loader.ObjectLoader import ObjectLoader
from src.utility.CameraUtility import CameraUtility
from src.utility.LightUtility import Light
from src.utility.MathUtility import MathUtility
from src.Eugene.dataset_utils import run_modules,sample_camera_pose

from src.utility.RendererUtility import RendererUtility
from src.utility.PostProcessingUtility import PostProcessingUtility
from src.utility.Utility import Utility
import numpy as np
from src.utility.filter.Filter import Filter
from src.utility.MeshObjectUtility import MeshObject
import bpy
import matplotlib.pyplot as plt



Initializer.init()

# load the objects into the scene
# LOAD THE SCENE
scene = BlendLoader.load('examples/assets/scenes/room.blend')

# LOAD THE OBJECT INTO THE SCENE
object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ModelNetLoader",
        "config": {
          "data_path": 'resources/ModelNet40',
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

# define a light and set its location and energy level
# light = Light()
# light.set_type("POINT")
# light.set_location([5, -5, 5])
# light.set_energy(1000)

# define the camera intrinsics
# CameraUtility.set_intrinsics_from_blender_params(1, 512, 512, lens_unit="FOV")

intrinsics = np.array([[128., 0.,63.5],
                       [0.,126.7, 63.5 ],
                       [0.,0.,1.]])

CameraUtility.set_intrinsics_from_K_matrix(intrinsics, image_width = 128, image_height = 128,clip_start=0.001)

# read the camera positions file and convert into homogeneous camera-world transformation

for _ in range(2):
        pos_volume = ([-0.04, -0.04, 0.15],
                      [0.04, 0.04, 0.25])  # ([0.,0.,0.15],[0.,0.,0.15]) #  ([-0.03, -0.03, 0.13], [0.03, 0.03, 0.16]) #
        orn_volume = ([0., 0., 0.], [0., 0., np.pi * 2])

        T_wc = sample_camera_pose(pos_volume, orn_volume, pre_selected_visible_objects=Filter.by_cp(
            elements=MeshObject.convert_to_meshes(bpy.data.objects), cp_name='manip_object', value=True),
                                  fully_visible=True)
        CameraUtility.add_camera_pose(T_wc)

# activate normal and distance rendering
RendererUtility.enable_normals_output()
RendererUtility.enable_distance_output()
# set the amount of samples, which should be used for the color rendering
RendererUtility.set_samples(350)

# render the whole pipeline
data = RendererUtility.render()

a = data['normals']
print('')
