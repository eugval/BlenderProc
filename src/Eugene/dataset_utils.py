import bpy
from src.utility.Utility import Utility
import numpy as np
from src.utility.MeshObjectUtility import MeshObject
from src.utility.lighting.SurfaceLighting import SurfaceLighting
import colorsys
from src.Eugene import se3_tools as se3
from src.utility.camera.CameraValidation import CameraValidation
from src.utility.CameraUtility import CameraUtility
from src.utility.EntityUtility import  Entity as EntityUtility
from src.utility.filter.Filter import Filter
from src.utility.MathUtility import MathUtility
from src.utility.LightUtility import Light
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.utility.RendererUtility import RendererUtility
import copy

import pyfastnoisesimd as fns
import skimage.morphology as imorph




def run_modules(modules):
    for module in modules:
        module.run()

def fix_uv_maps():
    uv_fix_loading_module = Utility.initialize_modules([
        {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "type": "MESH"
                    }
                },
                "cf_add_uv_mapping": {
                    "projection": "cube",
                    "forced_recalc_of_uv_maps": True
                }
            }
        }
    ])
    run_modules(uv_fix_loading_module)


def loguniform_sampling(min,max, size = None):
    return np.asarray(np.exp(np.random.uniform(np.log(min), np.log(max), size)))

def sample_color(h_range, s_range, v_range, rgba = False):
    h = np.random.uniform(*h_range)
    s = np.random.uniform(*s_range)
    v = np.random.uniform(*v_range)

    rgb = colorsys.hsv_to_rgb(h,s,v)
    if(rgba):
        return list(rgb)+[1.0]
    else:
        return list(rgb)



def cam_from_blender_to_opencv(blender_cam_posemat):
    new_pose = MathUtility.change_source_coordinate_frame_of_transformation_matrix(blender_cam_posemat,  ["X", "-Y", "-Z"])
    return new_pose


def sample_camera_pose(pos_volume, orn_volume, objects_visible = [], pre_selected_visible_objects =[] ,validation_threshold  = 6000,
                       fully_visible=True, obj_name_check= None,
                       relative_to_top=False):
    '''
    Samples a camear pose within the specified volume, making sure the specified objects are fully visible in the image.
    objects_visible : List of object names to be fully visible.
    pre_selected_visible_objects : List of blenderproc Meshobjects to be fully visible.
    '''
    validated = False
    count = 0
    while not validated:
        count +=1

        if count % 50 == 0:
            max_x = pos_volume[1][0]
            new_max_x = (max_x - 0.01)
            pos_volume[1][0] = new_max_x if new_max_x >0. else 0.
            pos_volume[0][0] = - new_max_x if new_max_x >0. else 0.

            max_y = pos_volume[1][1]
            new_max_y = (max_y - 0.01)
            pos_volume[1][1] = new_max_y if new_max_y > 0. else 0.
            pos_volume[0][1] = - new_max_y if new_max_y > 0. else 0.

            if(max_x < 0.04 or max_y<0.04):
                pos_volume = copy.deepcopy(pos_volume)
                max_z = pos_volume[1][2]
                new_max_z = max_z + 0.01
                pos_volume[1][2] = new_max_z if new_max_z < 0.85 else 0.85


        if(count>validation_threshold):
            raise TimeoutError('Cannot find a camera pose with the obects specified fully visible.')

        pos = np.random.uniform(*pos_volume)
        orn = np.random.uniform(*orn_volume)
        rot = se3.euler2rot('XYZ', orn)
        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3, :3] = rot

        all_objs  = MeshObject.convert_to_meshes(bpy.data.objects)
        selected_objs = [Filter.one_by_attr(all_objs,'name', obj_name) for obj_name in objects_visible]
        selected_objs += pre_selected_visible_objects

        assert len(selected_objs) == 1
        if(obj_name_check is not None):
            assert selected_objs[0].get_name() == obj_name_check


        for obj in selected_objs:
            # obj = Filter.one_by_attr(all_objs,'name', obj_name)
            bb = np.asarray(obj.get_bound_box())


            if(relative_to_top):
                #TODO: Note this only makes sense with a single selected object
                mean_top_point = obj.get_top_mean_point()
                z_top_point = mean_top_point[2]
                pose[2,3] += z_top_point


            if(not fully_visible):
                # If not fully visible, only the centrepoint needs ot be in in image, otherwise, all 8 corners need to be in image
                bb = np.mean(bb, axis= 0, keepdims =True)


            bb = np.concatenate([bb, np.ones((bb.shape[0], 1))], axis=1)

            # cam_ob = bpy.context.scene.camera
            camera_pose_in_world = cam_from_blender_to_opencv(pose)

            # project bounding box in image plane
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = camera_pose_in_world[:3, :3].T
            extrinsics[:3, 3] = - camera_pose_in_world[:3, :3].T @ np.ascontiguousarray(camera_pose_in_world[:3, 3])

            intrinsics = np.asarray(CameraUtility.get_intrinsics_as_K_matrix())

            cam2_frame_coords = extrinsics.dot(bb.T)
            cam2_intrinsics = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
            cam2_image_coords = cam2_intrinsics.dot(cam2_frame_coords)
            cam2_image_coords /= cam2_image_coords[-1, :]
            cam2_image_coords = cam2_image_coords[:2, :]

            width = bpy.context.scene.render.resolution_x
            height = bpy.context.scene.render.resolution_y
            row0_selection = np.logical_and(cam2_image_coords[0, :] < width, cam2_image_coords[0, :] >= 0)
            row1_selection = np.logical_and(cam2_image_coords[1, :] < height, cam2_image_coords[1, :] >= 0)
            row_selection = np.logical_not(np.logical_and(row0_selection, row1_selection))

            if (not row_selection.any()):
                validated = True

    return pose





def sample_camera_pose_delta(initial_pose, pos_volume, orn_volume, objects_visible = [], pre_selected_visible_objects =[] ,validation_threshold  = 6000,
                       fully_visible=True, obj_name_check= None):
    '''
    Samples a camear pose within the specified volume, making sure the specified objects are fully visible in the image.
    objects_visible : List of object names to be fully visible.
    pre_selected_visible_objects : List of blenderproc Meshobjects to be fully visible.
    '''
    validated = False
    count = 0
    while not validated:
        count +=1

        if count % 50 == 0:
            max_x = pos_volume[1][0]
            new_max_x = (max_x - 0.01)
            pos_volume[1][0] = new_max_x if new_max_x >0. else 0.
            pos_volume[0][0] = - new_max_x if new_max_x >0. else 0.

            max_y = pos_volume[1][1]
            new_max_y = (max_y - 0.01)
            pos_volume[1][1] = new_max_y if new_max_y > 0. else 0.
            pos_volume[0][1] = - new_max_y if new_max_y > 0. else 0.

            if(max_x < 0.04 or max_y<0.04):
                pos_volume = copy.deepcopy(pos_volume)
                max_z = pos_volume[1][2]
                new_max_z = max_z + 0.01
                pos_volume[1][2] = new_max_z if new_max_z < 0.85 else 0.85


        if(count>validation_threshold):
            raise TimeoutError('Cannot find a camera pose with the obects specified fully visible.')

        pos = np.random.uniform(*pos_volume)
        orn = np.random.uniform(*orn_volume)
        rot = se3.euler2rot('XYZ', orn)
        # cam_rot_delta = se3.so3_exp(np.array([0., 0., 3.14]).astype(float))
        cam_pose_delta = se3.make_pose(pos, rot)
        pose = np.matmul(cam_pose_delta, initial_pose)



        all_objs  = MeshObject.convert_to_meshes(bpy.data.objects)
        selected_objs = [Filter.one_by_attr(all_objs,'name', obj_name) for obj_name in objects_visible]
        selected_objs += pre_selected_visible_objects

        assert len(selected_objs) == 1
        if(obj_name_check is not None):
            assert selected_objs[0].get_name() == obj_name_check


        for obj in selected_objs:
            # obj = Filter.one_by_attr(all_objs,'name', obj_name)
            bb = np.asarray(obj.get_bound_box())


            if(not fully_visible):
                # If not fully visible, only the centrepoint needs ot be in in image, otherwise, all 8 corners need to be in image
                bb = np.mean(bb, axis= 0, keepdims =True)


            bb = np.concatenate([bb, np.ones((bb.shape[0], 1))], axis=1)

            # cam_ob = bpy.context.scene.camera
            camera_pose_in_world = cam_from_blender_to_opencv(pose)

            # project bounding box in image plane
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = camera_pose_in_world[:3, :3].T
            extrinsics[:3, 3] = - camera_pose_in_world[:3, :3].T @ np.ascontiguousarray(camera_pose_in_world[:3, 3])

            intrinsics = np.asarray(CameraUtility.get_intrinsics_as_K_matrix())

            cam2_frame_coords = extrinsics.dot(bb.T)
            cam2_intrinsics = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
            cam2_image_coords = cam2_intrinsics.dot(cam2_frame_coords)
            cam2_image_coords /= cam2_image_coords[-1, :]
            cam2_image_coords = cam2_image_coords[:2, :]

            width = bpy.context.scene.render.resolution_x
            height = bpy.context.scene.render.resolution_y
            row0_selection = np.logical_and(cam2_image_coords[0, :] < width, cam2_image_coords[0, :] >= 0)
            row1_selection = np.logical_and(cam2_image_coords[1, :] < height, cam2_image_coords[1, :] >= 0)
            row_selection = np.logical_not(np.logical_and(row0_selection, row1_selection))

            if (not row_selection.any()):
                validated = True

    return pose







def project_object_top_point(mesh_object, camera_pose):
    bb = mesh_object.get_top_mean_point()[None,: ]

    bb = np.concatenate([bb, np.ones((bb.shape[0], 1))], axis=1)

    # cam_ob = bpy.context.scene.camera
    camera_pose_in_world = cam_from_blender_to_opencv(camera_pose)

    # project bounding box in image plane
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = camera_pose_in_world[:3, :3].T
    extrinsics[:3, 3] = - camera_pose_in_world[:3, :3].T @ np.ascontiguousarray(camera_pose_in_world[:3, 3])

    intrinsics = np.asarray(CameraUtility.get_intrinsics_as_K_matrix())

    cam2_frame_coords = extrinsics.dot(bb.T)
    cam2_intrinsics = np.concatenate([intrinsics, np.zeros((3, 1))], axis=1)
    cam2_image_coords = cam2_intrinsics.dot(cam2_frame_coords)
    cam2_image_coords /= cam2_image_coords[-1, :]
    cam2_image_coords = cam2_image_coords[:2, :]

    return np.round(cam2_image_coords[:,0]).astype('int')



def set_lights(ambient_strength_range=[0.5,25], light_colour_ranges =[[0., 1.0], [0.2, 1.0], [0.4, 1.0]],
               ambient_light_colour = [1.0, 1.0, 1.0, 1.0] , light_location_volume = [[-3, -3, 0], [3, 3, 3]], energy_range = [30,450],
               light_colour = None, number_of_lights = 1):
    emission_strength_sample = loguniform_sampling(*ambient_strength_range)
    emission_color = ambient_light_colour
    ambient_light_plane = MeshObject(bpy.data.objects['light_plane'])
    SurfaceLighting.run([ambient_light_plane], emission_strength=emission_strength_sample,
                        emission_color=emission_color)
    # sample colour
    if (light_colour is not None):
        colour = light_colour
    else:
        colour = sample_color(*light_colour_ranges)

    created_lights = []
    for i in range(number_of_lights):
        # randomise directional light
        # sample location
        location = np.random.uniform(*light_location_volume)

        # sample energy
        energy = np.random.uniform(*energy_range)

        light = Light()
        light.set_type("POINT")
        light.set_location(location)
        light.set_color(colour)
        light.set_energy(energy)
        created_lights.append(light)



    return created_lights, colour


def randomise_light_position_colour_and_energy(lights, location_volume = None, colour_change_percent=0., energy_range= None, frame_num = None):
    if location_volume is None:
        location_volume = [[-1.5, -1.5, 0.02], [1.5, 1.5, 0.3]]



    h_change =np.random.uniform(-1,1)*colour_change_percent
    s_change = np.random.uniform(-1,1)*colour_change_percent
    v_change = np.random.uniform(-1,1)*colour_change_percent

    for light in lights:
        # Randomise location
        location = np.random.uniform(*location_volume)
        location +=  np.sign(location)*np.array([0.35,0.35,0.])
        light.set_location(location,frame = frame_num )

        #Randomise energy
        if(energy_range is not None):
            energy = np.random.uniform(*energy_range)
            light.set_energy(energy)

        #Randomise colour
        previous_h, previous_s, previous_v = colorsys.rgb_to_hsv(*light.get_color())
        new_h, new_s, new_v = np.clip(previous_h + h_change,0.,1.), np.clip(previous_s + s_change,0.,1.), np.clip(previous_v + v_change,0.,1.)
        new_rgb = colorsys.hsv_to_rgb(new_h, new_s,new_v)
        light.set_color(new_rgb, frame_num)


def min_max_stardardisation_to_01(image):
    image_min = np.min(image)
    image_max = np.max(image)
    range = (image_max-image_min)

    return (image-image_min)/range



def render_scene(activate_renderer = False):
    if(activate_renderer):
        RendererUtility.enable_normals_output()
        RendererUtility.enable_distance_output()
        RendererUtility.enable_depth_output()
        RendererUtility.set_samples(50)

    # render
    data = RendererUtility.render(load_keys={'colors', 'depth', 'normals', 'dist'})

    # Run Rendering Segmentation
    seg_data = SegMapRendererUtility.render(map_by=["class", "instance",  "name", "cp_manip_object"])

    name = Filter.one_by_cp(elements=MeshObject.convert_to_meshes(bpy.data.objects),
                                             cp_name='manip_object', value=True).get_name()
    seg_map_object = aggregate_segmap_values('name', name, seg_data)

    return data, seg_data, seg_map_object


def aggregate_segmap_values(key,value, seg_data):
    #TODO: extend to many seg maps and many attributes
    seg_map = seg_data['instance_segmaps'][0]
    attribute = seg_data['instance_attribute_maps'][0]

    indices_to_merge =set()

    for object in attribute:
        if (object[key]==value):
            indices_to_merge.add(object['idx'])

    seg_map_new = np.zeros_like(seg_map)
    for i in indices_to_merge:
        seg_map_new+=seg_map==i



    return seg_map_new



def resize_intrinsics(K, resize_ratio):
    '''
    When resizing an image, the intrinsics changes accordingly. Note: Note sure if this works when there is a
    distortion coefficient
    K: Intrinsics matrix
    resize_ratio: new_size/old_size for each of the x and y image dimensions. x goes in the 0'th index
    '''
    new_k = copy.deepcopy(K)
    new_k[0]*=resize_ratio[0]
    new_k[1]*=resize_ratio[1]
    return new_k



def process_bottleneck(num_frame, chunk_id, pos_volume = None, orn_volume = None, manip_object = None, writer= None):
    '''
    manip_object : BlenderProc Meshobject , string or None. if not none then the bottleneck pose will be calculated as a delta from the top point of the mani object.
                    if it is a string then it needs to be the name of the object it is referring to.
    '''
    # Place camera at bottleneck
    # Get object of interest

    top_point = np.zeros((3,))
    if (manip_object is not None):
        if (isinstance(manip_object, str)):
            manip_object = MeshObject(manip_object)
        elif(not isinstance(manip_object, MeshObject)):
            # Get the top point of its bounding box
            raise ValueError('Manip object needs to be string or MeshObject')
        top_point = manip_object.get_top_mean_point()


    # Sample the bottleneck pose
    if(pos_volume is None):
        pos_delta = np.random.uniform([-0.02, -0.02, 0.11], [0.02, 0.02, 0.13])
    else:
        pos_delta = np.random.uniform(*pos_volume)

    bottleneck_pos = top_point + pos_delta

    if(orn_volume is None):
        residual_rot = np.random.uniform([-0.2618 / 2, -0.2618 / 2, -0.7854], [0.2618 / 2, 0.2618 / 2, 0.7854])
    else:
        residual_rot =  np.random.uniform(*orn_volume)

    orientation = CameraUtility.rotation_from_forward_vec(np.array([0.,0.,-1.]), up_axis='Y', #top_point - bottleneck_pos
                                                          residual_rot=residual_rot)

    bottleneck_pose = np.eye(4)
    bottleneck_pose[:3, 3] = bottleneck_pos
    bottleneck_pose[:3, :3] = orientation
    CameraUtility.add_camera_pose(bottleneck_pose, 0)

    # render
    data, seg_data, seg_map_cups = render_scene(num_frame, chunk_id,bottleneck_pose, is_bottleneck =True, writer=writer)

    # return bottleneck stuff
    return bottleneck_pose, data['colors'][0], data['depth'][0], seg_map_cups


def  apply_twist(current_pose, twist, timestep =0.1):
    '''
    Applies the twist. twist is in the world frame so it is applied before the current pose.

    current_pose = Twc  Twist = T_delta in world
    so new pose = exp(Twis*time).dot(Twc)
    note the if the twist was in the ee frame then we would have post-multiplication.
    http://publish.illinois.edu/ece470-intro-robotics/files/2019/09/09-lecture.pdf
    '''
    delta_pose = twist * timestep

    delta_pose = se3.se3_exp(delta_pose)

    return current_pose.dot(delta_pose)



def superpose_bottleneck(current_image,bottleneck_image, bottleneck_seg_map):
    bottleneck_seg_map = bottleneck_seg_map[:,:,None]
    current_image = current_image*(1.-bottleneck_seg_map) + 0.5*bottleneck_seg_map*current_image + 0.2*bottleneck_image*bottleneck_seg_map + 0.3* np.array([0.,1.0,0.])*bottleneck_seg_map

    return current_image

def superpose_seg_mask(current_image,seg_mask):
    seg_mask = seg_mask[:,:,None]
    current_image = current_image*(1-seg_mask) +  0.7*seg_mask*current_image + 0.3* np.array([1.0,0.0,0.])*seg_mask

    return current_image


def light_setting_selection(light_setting_choice):
    if (light_setting_choice == 'mostly_ambient'):
        ambient_strength_range = [1.0, 20.0]
        light_location_volume = [[-4.0, -4.0, 0.5], [4.0, 4.0, 2.4]]
        number_of_lights = np.random.choice([1, 2])
        energy_range = [1, 200]
    elif (light_setting_choice == 'strong_top_shadow'):
        ambient_strength_range = [0.1, 2.0]
        light_location_volume = [[-1.2, -1.2, 1.3], [1.2, 1.2, 2.3]]
        energy_range = [500, 1000]
        number_of_lights = 1
    elif (light_setting_choice == 'generic_shadow'):
        ambient_strength_range = [0.1, 10.0]
        light_location_volume = [[-2., -2., 0.5], [2., 2., 2.4]]
        number_of_lights = np.random.choice([1, 2])
        energy_range = [50, 1000]
    elif (light_setting_choice == 'very_bright'):
        ambient_strength_range = [5.0, 20.0]
        light_location_volume = [[-1.5, -1.5, 0.5], [1.5, 1.5, 2.4]]
        number_of_lights = np.random.choice([1, 2])
        energy_range = [800, 1200]
    elif (light_setting_choice == 'very_dim'):
        ambient_strength_range = [0.1, 3.0]
        light_location_volume = [[-1.5, -1.5, 0.5], [1.5, 1.5, 2.4]]
        number_of_lights = np.random.choice([1, 2])
        energy_range = [50, 200]
    else:
        raise NotImplementedError()

    return ambient_strength_range,light_location_volume,number_of_lights,energy_range



def generate_perlin_noise(bitmat_shape, seed = 0):
    perlin = fns.Noise(seed=seed, numWorkers=4)

    perlin.noiseType = fns.NoiseType.SimplexFractal
    perlin.fractal.fractalType = fns.FractalType.FBM

    rndOct = np.random.choice([2, 3, 4], size=1)
    # rndOct = 8
    perlin.fractal.octaves = rndOct
    # perlin.fractal.lacunarity = 2.1
    perlin.fractal.lacunarity = np.random.uniform(2.0, 4.0)
    perlin.fractal.gain = 0.5

    perlin.perturb.perturbType = fns.PerturbType.NoPerturb

    perlin.frequency = np.random.uniform(0.005, 0.1)
    VecF0 = perlin.genAsGrid(bitmat_shape)
    perlin.frequency = np.random.uniform(0.005, 0.1)
    VecF1 = perlin.genAsGrid(bitmat_shape)

    Wxy = np.random.uniform(1, 12)
    X, Y = np.meshgrid(np.arange(bitmat_shape[1]), np.arange(bitmat_shape[0]))

    res = {
        'VecF0':VecF0,
        'VecF1':VecF1,
         'Wxy': Wxy,
        'X':X,
        'Y':Y
    }
    return res


def perlin_bitmap_deformation(bit_map, perlin_noise_parameters):
    bit_map = copy.deepcopy(bit_map)
    # perlin = fns.Noise(seed=seed, numWorkers=4)
    #
    # perlin.noiseType = fns.NoiseType.SimplexFractal
    # perlin.fractal.fractalType = fns.FractalType.FBM
    #
    # rndOct = np.random.choice([2, 3, 4], size=1)
    # # rndOct = 8
    # perlin.fractal.octaves = rndOct
    # # perlin.fractal.lacunarity = 2.1
    # perlin.fractal.lacunarity = np.random.uniform(2.0, 4.0)
    # perlin.fractal.gain = 0.5
    #
    # perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    #
    # perlin.frequency = loguniform_sampling( 0.001, 0.1)
    # VecF0 = perlin.genAsGrid(bit_map.shape)
    # perlin.frequency = loguniform_sampling( 0.001, 0.1)
    # VecF1 = perlin.genAsGrid(bit_map.shape)
    #
    # Wxy = np.random.uniform(1, 12)
    # X, Y = np.meshgrid(np.arange(bit_map.shape[1]), np.arange(bit_map.shape[0]))

    X = perlin_noise_parameters['X']
    Y = perlin_noise_parameters['Y']
    VecF0 = perlin_noise_parameters['VecF0']
    VecF1 = perlin_noise_parameters['VecF1']
    Wxy = perlin_noise_parameters['Wxy']

    # scale with depth
    fx = X + Wxy * VecF0
    fy = Y + Wxy * VecF1
    fx = np.where(fx < 0, 0, fx)
    fx = np.where(fx >= bit_map.shape[1], bit_map.shape[1] - 1, fx)
    fy = np.where(fy < 0, 0, fy)
    fy = np.where(fy >= bit_map.shape[0], bit_map.shape[0] - 1, fy)
    fx = fx.astype(dtype=np.uint16)
    fy = fy.astype(dtype=np.uint16)

    return bit_map[fy, fx]

def random_morphological_operation(bit_map):
    operation = np.random.choice(np.arange(3))

    if(operation == 0):
        return bit_map
    elif(operation ==1):
        return imorph.dilation(bit_map, imorph.square(3))
    elif(operation ==2):
        return imorph.erosion(bit_map, imorph.square(3))




def deform_segmentation(seg,environment_type, perlin_noise_parameters, extra_segmentation,  seed=0 ) :
    if (environment_type == 'perlin'):
        new_seg = perlin_bitmap_deformation(seg, perlin_noise_parameters)
        return new_seg
    elif('extra' in environment_type):
        new_seg = seg# perlin_bitmap_deformation(seg, perlin_noise_parameters)
        return np.clip(new_seg + extra_segmentation,0,1)

    elif(environment_type =='random'):
        seg = copy.deepcopy(seg)
        mis_segmentation_map = generate_random_mis_segmentation_map((64, 64),
                                                                num_segs=np.random.randint(0,4),
                                                                seg_size=np.random.randint(4, 16), seed=0)
        return np.clip(seg + mis_segmentation_map, 0, 1)


    else:
        new_seg = seg
        return new_seg



def deform_segmentation_bottleneck(seg,environment_type, perlin_noise_parameters, seed=0 ):
    if (environment_type == 'perlin'):
        new_seg = perlin_bitmap_deformation(seg, perlin_noise_parameters)
        return new_seg
    else:
        new_seg = seg
        return new_seg



def generate_mis_segmentation_map(shape, seg_points_x, seg_points_y, seg_size = 10, perlin_noise_params = None):
    seg_map = np.zeros(shape)
    half_seg_size = seg_size//2


    for x, y in zip(seg_points_x,seg_points_y):
        seg_map[x-half_seg_size:x+half_seg_size, y-half_seg_size:y+half_seg_size] = 1


    if(perlin_noise_params is not None):
        seg_map = perlin_bitmap_deformation(seg_map, perlin_noise_params)

    return seg_map


def generate_random_mis_segmentation_map(shape, num_segs = 1, seg_size = 10, seed = 0):
    half_seg_size = seg_size//2

    seg_points_x = np.random.randint(  half_seg_size+1, shape[1] - half_seg_size - 1,size=(num_segs,) )
    seg_points_y = np.random.randint( half_seg_size+1, shape[0] - half_seg_size-1,size=(num_segs,) )

    perlin_noise_params = generate_perlin_noise(shape, seed = seed)

    seg_map = generate_mis_segmentation_map(shape, seg_points_x, seg_points_y, seg_size = 10, perlin_noise_params= perlin_noise_params)

    return seg_map



