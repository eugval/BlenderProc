import bpy

from src.main.Module import Module
from src.utility.ItemCollection import ItemCollection
from src.utility.LightUtility import Light


class LightInterface(Module):
    """ 
    **Configuration**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - cross_source_settings
          - See the next table for which properties can be set. Default: {}.
          - dict

    **Properties per lights entry**:

    .. list-table:: 
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - location
          - The position of the light source, specified as a list of three values. Default: [0, 0, 0]
          - list
        * - rotation
          - The rotation of the light source, specified as a list of three euler angles. Default: [0, 0, 0]
          - list
        * - color
          - Light color, specified as a list of three values [R, G, B]. Default: [1, 1, 1]. Range: [0, inf]
          - list
        * - distance
          - Falloff distance of the light = point where light is half the original intensity. Default: 0. Range: [0, inf]
          - float
        * - energy
          - Intensity of the emission of a light source. Default: 10.
          - float
        * - type
          - The type of a light source. Default: POINT. Available: [POINT, SUN, SPOT, AREA]
          - string
    """

    def __init__(self, config):
        Module.__init__(self, config)
        self.cross_source_settings = self.config.get_raw_dict("cross_source_settings", {})
        self.light_source_collection = ItemCollection(self._add_light_source, self.cross_source_settings)

    def _add_light_source(self, config):
        """ Adds a new light source according to the given configuration.

        :param config: A configuration object which contains all parameters relevant for the new light source.
        """
        light = Light()

        light.set_type(config.get_string("type", 'POINT'))
        light.set_location(config.get_list("location", [0, 0, 0]))
        light.set_rotation_euler(config.get_list("rotation", [0, 0, 0]))
        light.set_energy(config.get_float("energy", 10.))
        light.set_color(config.get_list("color", [1, 1, 1])[:3])
        light.set_distance(config.get_float("distance", 0))

        if config.has_param('locations'):
            keyframe = 0
            for location in config.get_list("locations"):
                if bpy.context.scene.frame_end < keyframe + 1:
                    bpy.context.scene.frame_end = keyframe + 1
                light.set_location(list(location))
                light.blender_obj.keyframe_insert(data_path='location', frame=keyframe)
                keyframe+=1

        if config.has_param('colors'):
            keyframe = 0
            for color in config.get_list("colors"):
                if bpy.context.scene.frame_end < keyframe + 1:
                    bpy.context.scene.frame_end = keyframe + 1
                light.set_color(list(color[:3]))
                light.blender_obj.data.keyframe_insert(data_path='color', frame=keyframe)
                keyframe+=1


        if config.has_param('energies'):
            keyframe = 0
            for energy in config.get_list("energies"):
                if bpy.context.scene.frame_end < keyframe + 1:
                    bpy.context.scene.frame_end = keyframe + 1
                light.set_energy(float(energy))
                light.blender_obj.data.keyframe_insert(data_path='energy', frame=keyframe)
                keyframe+=1

