# This file can be used to execute the pipeline from the blender scripting UI
import os
import bpy
import sys
import pydevd_pycharm

#pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

# Make sure the current script directory is in PATH, so we can load other python modules
working_dir = os.path.dirname(bpy.context.space_data.text.filepath) + "/../"

if not working_dir in sys.path:
    sys.path.append(working_dir)

# Add path to custom packages inside the blender main directory
if sys.platform == "linux" or sys.platform == "linux2":
    packages_path = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "..", "..", "custom-python-packages"))
elif sys.platform == "darwin":
    packages_path = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "..", "..", "..", "Resources", "custom-python-packages"))
elif sys.platform == "win32":
    packages_path = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "..", "..", "custom-python-packages"))
else:
    raise Exception("This system is not supported yet: {}".format(sys.platform))
sys.path.append(packages_path)

# Delete all loaded models inside src/, as they are cached inside blender
for module in list(sys.modules.keys()):
    if module.startswith("src") or module.startswith("external"):
        del sys.modules[module]

from src.main.Pipeline import Pipeline


# Replace placeholders manually or use --debug command line argument
config_path = "examples/simple_coffee_mugs/config.yaml"
args = [ "examples/simple_coffee_mugs/output" , " resources/ModelNet40/", 1,
        ]  # Put in here arguments to use for filling the placeholders in the config file.


# Focus the 3D View, this is necessary to make undo work (otherwise undo will focus on the scripting area)
for window in bpy.context.window_manager.windows:
    screen = window.screen

    for area in screen.areas:
        if area.type == 'VIEW_3D':
            override = {'window': window, 'screen': screen, 'area': area}
            bpy.ops.screen.screen_full_area(override)
            break

# Store temp files in the same directory for debugging
temp_dir = "examples/simple_coffee_mugs/temp"

try:
    # In this debug case the rendering is avoided, everything is executed except the final render step
    # For the RgbRenderer the undo is avoided to have a direct way of rendering in debug
    pipeline = Pipeline(config_path, args, working_dir, temp_dir, avoid_output=True)
    pipeline.run()
finally:
    # Revert back to previous view
    bpy.ops.screen.back_to_previous()
