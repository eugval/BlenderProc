from src.utility.SetupUtility import SetupUtility
argv = SetupUtility.setup(['matplotlib','opencv-contrib-python', 'scikit-learn', 'numpy==1.20'  ,'numba', 'scipy', ]) #'pydevd-pycharm~=193.6911.2'

import sys
print(sys.argv)

from src.utility.Utility import Utility
from src.utility.CameraUtility import CameraUtility

from src.utility.WriterUtility import WriterUtility
from src.utility.Initializer import Initializer
from src.utility.loader.BlendLoader import BlendLoader
from src.utility.LightUtility import Light
from src.utility.MathUtility import MathUtility
from src.utility.MeshObjectUtility import MeshObject
from src.utility.lighting.SurfaceLighting import SurfaceLighting
from src.utility.RendererUtility import RendererUtility
from src.utility.SegMapRendererUtility import SegMapRendererUtility



import numpy as np
import os
import bpy
import colorsys
import matplotlib.pyplot as plt





Utility.working_dir = os.path.abspath(os.curdir) + "/"+ "examples/shapenet_dataset" + "/"


Initializer.init()

# LOAD THE SCENE
scene = BlendLoader.load('scene/room_with_lights.blend')

# LOAD THE OBJECT INTO THE SCENE
object_loading_module = Utility.initialize_modules([
 {
        "module": "loader.ShapenetLoader",
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

