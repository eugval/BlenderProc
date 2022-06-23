from src.Eugene.dataset_utils import run_modules, fix_uv_maps
from src.utility.Utility import Utility
import numpy as np
import pickle
def load_modelnet(model_name = None,):
    if( model_name is not None):
        object_loading_module = Utility.initialize_modules([
            {
                "module": "loader.ModelNetLoader",
                "config": {
                    "data_path": '../../resources/ModelNet40',  # other_objects ModelNet40
                    "model_name": model_name, #cube cup_0100
                    # "scale_path": "./per_category_scale.json",
                    "manual_scale": 0.15,
                    "train": True,
                    "samples": 1,
                    "replacement": False,
                    "add_properties": {
                        "cp_physics": True,
                        "cp_manip_object": True,
                        "cp_distractor": False,
                    },
                    "cf_set_shading": "SMOOTH"
                }
            },

        ])

    else:
        object_loading_module = Utility.initialize_modules([
         {
                "module": "loader.ModelNetLoader",
                "config": {
                  "data_path": '../../resources/ModelNet40', #other_objects ModelNet40
                  # "model_name": "cube", #cube cup_0100
                   # "scale_path": "./per_category_scale.json",
                    "manual_scale": 0.15,
                  "train": True,
                  "samples" : 1,
                  "replacement": False,
                  "add_properties": {
                    "cp_physics": True,
                    "cp_manip_object":True,
                      "cp_distractor": False,
                  },
                  "cf_set_shading": "SMOOTH"
                }
              },

        ])

    run_modules(object_loading_module)



def randomise_manip_object_colour(randomisation_probabilities):
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
            "mode": "once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": ["base_color", "roughness", "metallic", "specular", "anisotropic", "sheen",
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
            "randomisation_types": ["monochrome_random", "image_random"],  # "image_random"
            "randomisation_type_probabilities": randomisation_probabilities,  # [0.2,0.8][0.7,0.3],[0.5,0.5]
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    }, ]))


def randomise_room():
    cloth_colours_path = '/home/eugene/Projects/dense_correspondence_control/dense_correspondence_control/learning/data/for_colour_extraction/segcolours.pckl'
    cloth_colours = np.mean(np.array(pickle.load(open(cloth_colours_path, 'rb'))), axis=0) / 255.

    ### Randomise the walls
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
                "cf_set_base_color": [cloth_colours[2], cloth_colours[1], cloth_colours[0], 1.0],
                # cloth colours is brg
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
                "mode": "once_for_each",
                "number_of_samples": 1,
                "parameters_to_randomise": ["roughness", "metallic", "specular", "anisotropic", "sheen",
                                            "clearcoat"],
                "randomisation_probabilities": [0.5, 0.25, 0.25, 0.25, 0.15, 0.05],
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
        }
    ]))


def add_distractors(number_of_distractor, distractor_max_dist, distractor_offset_x_dist, distractor_offset_y_dist):
    ##### ADD DISTRACTORS


    distractor_loader = [{
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
                    "max": [distractor_max_dist + distractor_offset_x_dist,
                            distractor_max_dist + distractor_offset_y_dist,
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
    ]

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
            "mode": "once_for_each",
            "number_of_samples": 1,
            "parameters_to_randomise": ["base_color", "roughness", "metallic", "specular", "anisotropic", "sheen",
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
            "reference_texture_prob": 0.2,
            "store_reference_texture": False,
            "randomisation_types": ["monochrome_random", "image_random"],  # "image_random"
            "randomisation_type_probabilities": [0.2, 0.8],  # [0.35,0.65],
            "color_textures_path": '~/Projects/resources/textures',
            "gray_textures_path": '~/Projects/resources/gray_textures',
        }
    }, ]))

    object_loading_module = Utility.initialize_modules(distractor_loader)

    run_modules(object_loading_module)

