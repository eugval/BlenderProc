import numpy as np
import matplotlib.pyplot as plt

def generate_random_mis_segmentation_map(shape, num_segs = 1, seg_size = 10):
    half_seg_size =seg_size//2

    seg_points_x = np.random.randint(  half_seg_size+1, shape[1] - half_seg_size - 1,size=(num_segs,) )
    seg_points_y = np.random.randint( half_seg_size+1, shape[0] - half_seg_size-1,size=(num_segs,) )

    perlin_noise_params = None

    seg_map = generate_mis_segmentation_map(shape, seg_points_x, seg_points_y, seg_size = 10, perlin_noise_params= perlin_noise_params)


    print('done')



def generate_mis_segmentation_map(shape, seg_points_x, seg_points_y, seg_size = 10, perlin_noise_params = None):
    seg_map = np.zeros(shape)
    half_seg_size =seg_size//2


    for x, y in zip(seg_points_x,seg_points_y):
        seg_map[x-half_seg_size:x+half_seg_size, y-half_seg_size:y+half_seg_size] = 1

    return seg_map

if __name__ == '__main__':

    generate_random_mis_segmentation_map((64,64), 3, seg_size=10)