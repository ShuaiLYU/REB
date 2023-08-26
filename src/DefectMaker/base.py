import  random
import math
from PIL import Image, ImageFilter
import numpy as np
import cv2 as cv
# from DMlib import MaskTool

from enum import Enum
from copy import deepcopy

from typing import List
from typing import Union

"""

15:40    2023年5月28日（星期日） (GMT+8)
add SHAPE  for  DRAEM meothd that 

"""
class SalienMethod(Enum):
    NONE=1
    IMAGE_FIT=0
    SALIENCY_FIT=2
    SALIENCY_FIT_INTER=3
    SALIENCY_CONSTRAINT=4
    SALIENCY_CONSTRAINT_INTER=5

    NOISE_THRESAH=6
    
    
    
class ImageSize(object):               
    def __init__(self,img=None):
        if img is not None:
            img=np.array(img)
            self.w,self.h=img.shape[1],img.shape[0]
    def init_with_size(self,img_size:tuple):
        self.w,self.h=img_size[0],img_size[1]
        return self
        
    def get(self):
        return self.w,self.h
        
"""

including size and location information 
"""
class Bbox(object):
    
    def __init__(self,x1,y1,x2,y2,img_size:ImageSize):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.img_size=img_size

        
    def random_move(self):
        bbox_w, bbox_h= self.x2-self.x1,self.y2-self.y1
        img_w,img_h=self.img_size.get()
        x = int(random.uniform(0, img_w - bbox_w))
        y = int(random.uniform(0, img_h - bbox_h))
        self.x1,self.y1=x,y
        self.x2,self.y2=self.x1+bbox_w,self.y1+bbox_h
        return self    
    
    def to_mask(self,foreground=255):
        return Mask(self._bbox2mask([self.x1,self.y1,self.x2,self.y2],self.img_size.get(),foreground))
    

    def move_by_center(self,center):
        center_x,center_y=center[0],center[1]
        
        bbox_w, bbox_h= self.x2-self.x1,self.y2-self.y1
        x = center_x-bbox_w//2
        y = center_y-bbox_h//2
        
        img_w,img_h=self.img_size.get()
        
        x= min(max(x,0),img_w-bbox_w)
        y= min(max(y,0),img_h-bbox_h)
  
        self.x1,self.y1=x,y
        self.x2,self.y2=self.x1+bbox_w,self.y1+bbox_h
        
    
    def _bbox2mask(self,bbox, img_size, foreground=255):
        """
        bbox  : (x1,y1,x2,y2)
        img_size: (w,h)
        """
        mask = np.zeros((img_size[1], img_size[0]), np.uint8)
        x1, y1, x2, y2 = bbox
        # print(bbox)
        mask[y1:y2, x1:x2] = foreground
        return mask
    
    def get(self):
        return [self.x1,self.y1,self.x2,self.y2]


class Mask(object):
    
    def __init__(self,array:np.ndarray,fill=255):
        self.fill=fill
        self.array= np.where(np.array(array)>0,fill,0).astype(np.uint8)

    
        
    def intersection(self,mask2):
        assert(self.array.shape== mask2.array.shape)
        return Mask(self.array*mask2.array,self.fill)
    
    def union(self,mask2):
        assert(self.array.shape== mask2.array.shape)
        return Mask(self.array+mask2.array,self.fill)
    
    
    def getRandomPoint(self):
        points=list(zip(*np.nonzero(self.array)))
        point=random.choice(points)
        return (point[1],point[0]) #(y,x) -> (x,y)
    
    def to_bbox(self):
        Rows, Cols = np.nonzero(self.array)
        x1, y1 = np.min(Cols), np.min(Rows)
        x2, y2 = np.max(Cols)+1, np.max(Rows)+1
        
        return Bbox(x1,y1,x2,y2,ImageSize(self.array))
    
    def crop(self,bbox:Bbox):
        x1,y1,x2,y2=bbox.get()
        return Shape(self.array[y1:y2,x1:x2],unpad=False)
    def get(self):
        return self.array
    
    def get_normalized(self,blur_ksize=None):
        
        normalize_img=Image.fromarray(self.array)
        if blur_ksize is not None:
            normalize_img = normalize_img.filter(ImageFilter.BoxBlur(blur_ksize))
            # print(normalize_img.size,blur_ksize)
        return np.array(normalize_img).astype(np.float32)/self.fill

    
class Saliency(Mask):
    def __init__(self,array:np.ndarray,fill=255):
        self.fill=fill
        self.array= np.where(np.array(array)>0,fill,0).astype(np.uint8)
    
                    
class Shape(object):
    
    def __init__(self,mask:[Mask,np.ndarray],unpad=True):
        if isinstance(mask,Mask):mask=mask.array
        if unpad: mask=self._from_mask(mask)
        self.array=mask.astype(np.bool)
    
    def _from_mask(self,array):
        """
        move the mask to origin of the coord
        """
        Rows, Cols = np.nonzero(array)
        x1, y1 = np.min(Cols), np.min(Rows)
        x2, y2 = np.max(Cols)+1, np.max(Rows)+1
        return array[y1:y2, x1:x2]

    def to_mask(self,bbox:Bbox=None,fill=255):
        if bbox is None:
            return  Mask(self.array,fill)
        else:
            img_size=bbox.img_size.get()
            array = np.zeros((img_size[1], img_size[0]), np.uint8)
            array[bbox.y1:bbox.y2,bbox.x1:bbox.x2]=self.array      
            return Mask(array,fill)
    
    
    def get(self):
        return self.array
    def get_wh(self):
        return self.array.shape[1],self.array.shape[0]    
                                 
        
    def intersection(self,shape2):
        assert(self.array.shape== shape2.array.shape)
        return Shape(self.array*shape2.array,unpad=False)
    
    def union(self,shape2):
        assert(self.array.shape== shape2.array.shape)
        return Shape(self.array+shape2.array,unpad=False)
 
    
class Instance(object):

    def __init__(self,bbox:Bbox=None,shape:Shape=None,patch=None,saliency:Saliency=None,salien_method:SalienMethod=SalienMethod.IMAGE_FIT,**kwargs):
        self.bbox=bbox
        self.shape=shape
        self.patch=patch
        self.saliency=saliency
        
        
        
    def random_move(self):
        if self.saliency is not None:
            random_center=self.saliency.getRandomPoint()
            self.bbox.move_by_center(random_center)
        else:
            self.bbox.random_move()
        
    def get_imshow_mask(self,fg=255,factor=None):
        mask=self.shape.to_mask(self.bbox,fg)
        return mask
    
############################################################################################

class CurveShape(Shape):
    def __init__(self,curve_mask:np.ndarray,r=0):
        curve_mask=self._rotate_mask(curve_mask,r)
        super(CurveShape,self).__init__(curve_mask)
    def _rotate_mask(self,curve:np.ndarray,r=0):
        if r!=0:
            curve=Image.fromarray(curve)
            curve=np.array(curve.rotate(r, expand=True) )
        return Mask(curve)
    

class RectShape(Shape):
    def __init__(self,w,h,r=0):
        mask=self._get_mask(w,h,r)
        super(RectShape,self).__init__(mask)
    def _get_mask(self,w,h,r=0):
        array=Image.new("L",(w,h),color=255)
        if r!=0:
            array=array.rotate(r, expand=True) 
        return Mask(np.array(array))
    
    
    
def get_image_channel(img:Image):
    assert(img.mode=="L" or img.mode=="RGB")
    return len(img.getbands())
        
        
class DmDataBase(object):
    
    def __init__(self,dataset,**kwargs):
        
        
        def get_one(idx):
            return dataset[idx]
        
        func_combs=[ functools.partial(get_one, idx=idx) for idx in range(len(dataset))  ]

        self.func_combs =func_combs
    

    
    def __getitem__(self,idx):
        func=self.func_combs[idx]
        return func
    
    def get_one(self,idx,**kwargs):
        # print(kwargs)
        func=self.func_combs[idx]
        return func(**kwargs)
    
    def __len__(self):
        return len(self.func_combs)
    
###############################################################################################


# class FillDataBase(object):
    
#     def __init__(self,dataset,**kwargs):
        
        
#         def get_one(idx):
#             return dataset[idx]
        
#         func_combs=[ functools.partial(get_one, idx=idx) for idx in range(len(dataset))  ]

#         self.func_combs =func_combs
    
    
#     def __getitem__(self,idx,**kwargs):
#         func=self.func_combs[idx]
#         return func(**kwargs)
    
#     def __len__(self):
#         return len(self.func_combs)
    
    

# import functools
# class RandomNoiseFill(FillDataBase):
    
    
#     def __init__(self,img_size,img_c, mean_range=[50,200],mean_step=10,fluct_range=[0,50],fluect_step=5,scale_range=[0,3], **kwargs):
        
    
#         mean_set=set(range(*mean_range,mean_step))
#         fluect_set=set(range(*fluct_range,fluect_step))
#         scale_set = set( [ 2**s for s in range(*scale_range)  ] + [ 1/(2**s) for s in range(*scale_range)   ])
#         print(scale_set)
        
#         func_combs=[]
#         for scale in scale_set:
#             for mean in mean_set:
#                 for fluct in fluect_set:
#                     param_comb={"img_size":img_size,"img_c":img_c,"mean":mean,"fluct":fluct,"scale":scale} 
#                     func= functools.partial(self.gen_random_noise_img, **param_comb) 
#                     func_combs.append(func)
#         self.func_combs =func_combs
      
            
#     @staticmethod
#     def gen_random_noise_img(img_size,img_c=1,mean=127,fluct=10,scale=1):
#         assert(len(img_size)==2)
#         # mean = random.randint(*mean)
#         # fluct = random.randint(*fluct)
#         low = max(mean - fluct ,0)
#         high = min(mean + fluct+1  ,255)
        
#         img_size= ( int(img_size[0]*scale), int(img_size[0]*scale) )
#         shape=(img_size[1],img_size[0]) if img_c==1 else (img_size[1],img_size[0],img_c)
#         defect = np.random.randint(low, high, shape)
#         return Image.fromarray(defect.astype(np.uint8)).resize((50,50))
    
    
    
    
    
###############################################################################################