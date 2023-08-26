

from torchvision import  transforms
import numbers
import  collections
import random
from torchvision.transforms import functional as F
import torch
from PIL import Image
import  numpy as np
import math
class  RandomTranspose(object):
    def __init__(self,params=None):
        if params is None:
            params=[None,Image.ROTATE_90 ,Image.ROTATE_180 ,Image.ROTATE_270,Image.TRANSPOSE ,Image.TRANSVERSE ,Image.TRANSVERSE,Image.FLIP_LEFT_RIGHT,Image.FLIP_LEFT_RIGHT]
        if not isinstance(params, list):
            raise ValueError("it must be a list.")
        self.params=params
    @staticmethod
    def get_params(params):
        param=random.choice(params)
        return param
    def transpose(self,img ,param):
        
        if param is None or img is None:
            return img
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        return  img.transpose(method=param)
    def __call__(self, img):
        param =self.get_params(self.params)
        
        if isinstance(img,list):
            return [self.transpose(item,param) for item in img ]
        elif isinstance(img,tuple):
            return [self.transpose(item,param) for item in img ]
        else:
            return self.transpose(img,param)



class RandomResizedCrop(transforms.RandomResizedCrop):
    
    
    
    def perform(self,img,i,j,h,w):
        if img is None:
            return None
        else:
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
  
        
        if isinstance(img,(list,tuple)):
            i, j, h, w = self.get_params(img[0], self.scale, self.ratio)
            return [ self.perform(item, i, j, h, w) for item in img ]
        else:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            return  self.perform(img, i, j, h, w)
