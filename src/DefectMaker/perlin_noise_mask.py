
"""
https://github.com/VitjanZ/DRAEM/blob/main/perlin.py

"""




def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        # print(grid.shape)
        angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

        tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
        dot = lambda grad, shift: (
                    np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                                axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])
    
def normalzie(array):
    array=array-np.min(array)
    array=array/np.max(array)
    return array

def binary_mask(array,threshold):
    return np.where(array > threshold, 1, 0)



import cv2
def resize(array,size):

    array=img = cv2.resize(array,size,interpolation=cv2.INTER_NEAREST)
    return array
    

    
    
import random
import numpy as np
import math

def log2foor(x):
    # print(x)
    return 2**math.ceil(math.log(x,2))
    
    
class PerlinNoise(object):
    
    def __init__(self,perlin_scale_range):

        self.perlin_scale_range=perlin_scale_range

    def get_one_mask(self,size,threshold=0.5,revise=False):
        
        min_perlin_scale,perlin_scale=self.perlin_scale_range
        self.perlin_scale_xy=perlin_scale_xy=(2**random.randint(min_perlin_scale,perlin_scale),2**random.randint(min_perlin_scale,perlin_scale))
        
        truncate_size=( log2foor(size[0]),log2foor(size[1])) 
        perlin_noise = rand_perlin_2d_np((truncate_size[1],truncate_size[0]), self.perlin_scale_xy)
        
        perlin_noise=normalzie(perlin_noise)
        if revise:
            perlin_noise=1-perlin_noise
        
        # print(np.array(perlin_noise).shape)
        mask=binary_mask(perlin_noise,threshold).astype(np.uint8)
        return  mask[:size[1],:size[0]]
    
    
    
def gen_perline_noise_mask(size,perlin_scale_xy,threshold=0.5,revise=False):

    truncate_size=( log2foor(size[0]),log2foor(size[1])) 
    # print(truncate_size)
    perlin_noise = rand_perlin_2d_np((truncate_size[1],truncate_size[0]), perlin_scale_xy)

    perlin_noise=normalzie(perlin_noise)
    if revise:
        perlin_noise=1-perlin_noise

    # print(np.array(perlin_noise).shape)
    mask=binary_mask(perlin_noise,threshold).astype(np.uint8)
    return  mask[:size[1],:size[0]]


import functools

from .base import *
from .perlin_noise_mask import gen_perline_noise_mask

"""

old perline shape
"""

class PerlineShapeMaker(DmDataBase):

    def __init__(self, perlin_scale_range=[0, 3], **kwargs):

        perlin_scale_x_set = set([2 ** scale for scale in list(np.arange(*perlin_scale_range, 1))])
        perlin_scale_y_set = set([2 ** scale for scale in list(np.arange(*perlin_scale_range, 1))])
        threshold_set = set(np.arange(*[0.5, 0.8], 0.1))
        revise_set = set([False])
        func_combs = []
        for perlin_scale_x in perlin_scale_x_set:
            for perlin_scale_y in perlin_scale_y_set:
                for threshold in threshold_set:
                    for revise in revise_set:
                        param_comb = {"perlin_scale_xy": (int(perlin_scale_x), int(perlin_scale_y)), "revise": revise,
                                      "threshold": threshold}
                        func = functools.partial(self.gen_perline_noise_mask, **param_comb)
                        func_combs.append(func)

        self.func_combs = func_combs

    @staticmethod
    def gen_perline_noise_mask(img_size, perlin_scale_xy, threshold=0.5, revise=False):
        # print(img_size,perlin_scale_xy,threshold,revise)

        return gen_perline_noise_mask(img_size, perlin_scale_xy, threshold, revise)

    def get_one_shape(self, img_size):
        idx = random.randint(0, len(self) - 1)
        shape_array = self.get_one(idx, img_size=img_size)
        return Shape(shape_array, unpad=False)


