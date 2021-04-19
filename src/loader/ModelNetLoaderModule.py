from src.loader.LoaderInterface import LoaderInterface
from src.utility.Utility import Utility
from src.utility.loader.ObjectLoader import ObjectLoader
import os
import random
import glob
import bpy
import json

class ModelNetLoaderModule(LoaderInterface):
    """ Just imports the objects for the given file path

    The import will load all materials into cycle nodes.

    **Configuration**:

    .. list-table::
        :widths: 25 100 10
        :header-rows: 1

        * - Parameter
          - Description
          - Type
        * - data_path
          - The path to the ModelNet base folder.
          - string
        * - category
          - A string of the modelnet category to use
          - string
        * - categories
          - A list of categories of objects to be sampled.
          - list of string
        * - train
          - A bool indicating whether to sample from the training models , default: True
          - bool
        * - model_name
          - the name of the model to load, needs to be in teh format of <category>_<model_number> with <model_nubmber> a
          4 digit integer.
          - str
        * - samples
          - Number of objects to load from the dataset. If model_name is set and replacement is True, then the same model
          loaded multiple number of times. default: 1
          - string
        * - replacement
          - A bool indicating whether to sample the models in the catergory with replacement. default: False
          - bool
    """
    def __init__(self, config):
        LoaderInterface.__init__(self, config)

    def run(self):
        if not self.config.has_param('data_path'):
            raise Exception("Need the path to be set")

        if(self.config.has_param('category') and self.config.has_param('categories')):
            raise Exception("Either set category or categories, but not both.")

        modelnet_path = self.config.get_string('data_path')
        modelnet_path = Utility.resolve_path(modelnet_path)


        if (not os.path.exists(modelnet_path)):
            raise Exception("ModelNet Path incorrect")


        train = self.config.get_bool('train', True)
        train_str = 'train' if train else 'test'

        samples = self.config.get_int('samples', 1)
        replacement = self.config.get_bool('replacement', False)


        # the file paths are mapped here to object names
        cache_objects = {} if not replacement else None
        loaded_objects = []
        if(self.config.has_param('model_name')):
            model_name =  self.config.get_string('model_name')
            model_id = model_name+'.off'

            category =  model_name.split('_')[0]
            category_path = os.path.join(modelnet_path, category)

            file_path = os.path.join(category_path,train_str,model_id)
            if(not os.path.exists(file_path)):
                raise  Exception("model {} does not exist in {}".format(model_id,os.path.join(category_path,train_str) ))

            for i in range(samples):
                current_objects =ObjectLoader.load(filepath=file_path, cached_objects=cache_objects)
                [obj.set_cp("category_id", category) for obj in current_objects]
                loaded_objects.extend(current_objects )

        elif(self.config.has_param('category')):
            category = self.config.get_string('category')
            category_path = os.path.join(modelnet_path, category)

            files = glob.glob(os.path.join(category_path,train_str,"*.off"))

            if replacement:
                selected_files = random.choices(files, k=samples)
            else:
                selected_files = random.sample(files, k=samples)

            for file_path in selected_files:
                current_objects = ObjectLoader.load(filepath=file_path, cached_objects=cache_objects)
                [obj.set_cp("category_id", category) for obj in current_objects]
                loaded_objects.extend(current_objects)

        elif (self.config.has_param('categories')):
            files  = []
            for category in self.config.get_list('categories'):
                #sample the categories, making sure that the different categories are balanced in their representation
                files_in_category = glob.glob(os.path.join(modelnet_path, category, train_str, "*.off"))
                len_files_in_category = len(files_in_category)
                files += random.sample(files_in_category, k = min(len_files_in_category, 2*samples))

            if replacement:
                selected_files = random.choices(files, k=samples)
            else:
                selected_files = random.sample(files, k=samples)

            for file_path in selected_files:
                category = file_path.split('/')[-1].split('_')[0]
                current_objects = ObjectLoader.load(filepath=file_path, cached_objects=cache_objects)
                [obj.set_cp("category_id", category) for obj in current_objects]
                loaded_objects.extend(current_objects)

        else:
            files = []
            for entry in os.listdir(modelnet_path):
                if not os.path.isdir(entry):
                   continue
                # sample the categories, making sure that the different categories are balanced in their representation
                files_in_category =  glob.glob(os.path.join(entry, train_str, "*.off"))
                len_files_in_category = len(files_in_category)
                files += random.sample(files_in_category, k=min(len_files_in_category, 2 * samples))

            if replacement:
                selected_files = random.choices(files, k=samples)
            else:
                selected_files = random.sample(files, k=samples)

            for file_path in selected_files:
                category = file_path.split('/')[-1].split('_')[0]
                current_objects = ObjectLoader.load(filepath=file_path, cached_objects=cache_objects)
                current_objects = [obj.set_cp("category_id", category) for obj in current_objects]
                loaded_objects.extend(current_objects)

        if not loaded_objects:
            raise Exception("No objects have been loaded here, check the config.")

        if(self.config.has_param('scale_path')):
            path = Utility.resolve_path(self.config.get_string('scale_path'))
            category_scale_dict = json.load(open(path,"rb"))
        else:
            category_scale_dict = None

        bpy.ops.object.select_all(action='DESELECT')
        for obj in loaded_objects:
            category = obj.get_cp("category_id")
            if(category_scale_dict is None):
                s_value = 1.0
            else:
                if category in category_scale_dict:
                    s_value = random.uniform(*category_scale_dict[category])
                else:
                    s_value = random.uniform(*category_scale_dict['fallback'])
            # obj.select()
            # bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')
            bpy.context.view_layer.objects.active = obj.blender_obj
            bb = obj.get_bound_box()
            diagonal = bb[-2] - bb[0]
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.transform.resize(value=[s_value / diagonal.length, s_value / diagonal.length, s_value / diagonal.length])
            bpy.ops.object.mode_set(mode='OBJECT')

            obj.move_origin_to_bottom_mean_point()




            # bb = obj.get_bound_box()
            # diagonal = bb[-2] - bb[0]
            # setattr(obj.blender_obj, 'scale', [1.0/diagonal.length] * 3)
            # print('diagonal legnth {}'.format(diagonal.length))



        bpy.ops.object.select_all(action='DESELECT')


        # Set the add_properties of all imported objects
        self._set_properties(loaded_objects)


