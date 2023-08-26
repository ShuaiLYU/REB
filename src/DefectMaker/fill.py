
from PIL import Image
import random
import numpy as np
import functools

from .base import *
from .perlin_noise_mask import gen_perline_noise_mask
from .bezier_mask import BezierMaskGenerator,gen_bbezier_mask

from .perlin_noise_mask import PerlinNoise
    

class RandomNoiseFillMaker(DmDataBase):
    
    
    def __init__(self,mean_range=[50,200],mean_step=10,fluct_range=[0,50],fluect_step=5,scale_range=[0,3],aspect_ratio_range=[0,3], **kwargs):
        
    
        mean_set=set(range(*mean_range,mean_step))
        fluect_set=set(range(*fluct_range,fluect_step))
        scale_set = set( [ 2**s for s in range(*scale_range)  ] )
        # scale_set = set( [ 2**s for s in range(*scale_range)  ] + [ 1/(2**s) for s in range(*scale_range)   ])
        aspect_ratio_set = set( [ 2**s for s in range(*aspect_ratio_range)  ] + [ 1/(2**s) for s in range(*aspect_ratio_range)   ])
        # print(scale_set)
        
        func_combs=[]
        for scale in scale_set:
            for mean in mean_set:
                for fluct in fluect_set:
                    for aspect_ratio in aspect_ratio_set:
                        # param_comb={"img_size":img_size,"img_c":img_c,"mean":mean,"fluct":fluct,"scale":scale} 
                        param_comb={"mean":mean,"fluct":fluct,"scale":scale,"aspect_ratio":aspect_ratio} 
                        func= functools.partial(self.gen_random_noise_img, **param_comb) 
                        func_combs.append(func)
        self.func_combs =func_combs
      
            
    @staticmethod
    def gen_random_noise_img(img_size,img_c=1,mean=127,fluct=10,scale=1,aspect_ratio=1):
        assert(len(img_size)==2)
        # mean = random.randint(*mean)
        # fluct = random.randint(*fluct)
        low = max(mean - fluct ,0)
        high = min(mean + fluct+1  ,255)
        
        scale_img_size= ( max(int(img_size[0]/scale*aspect_ratio),1),max(int(img_size[0]/scale/aspect_ratio),1))
        shape=(scale_img_size[1],scale_img_size[0]) if img_c==1 else (scale_img_size[1],scale_img_size[0],img_c)
        defect = np.random.randint(low, high, shape)
        return Image.fromarray(defect.astype(np.uint8)).resize(img_size)
    
    
    def get_one_fill(self,img_size,img_c=1):
        idx=random.randint(0,len(self)-1)
        return self.get_one(idx,img_size=img_size,img_c=img_c)
    

        
        
"""
从图像中随机crop一个图像，作为fill， 
可传入shape参数

"""

class CutFillMaker(object):
    
    def __init__(self,dataset,saliency_method:SalienMethod=SalienMethod.SALIENCY_CONSTRAINT):
        self.dataset=dataset
        self.saliency_method=saliency_method
    def get_one_fill(self,fill_size,img_c=1,**kwargs):
    
        sample=self.dataset[random.randint(0,len(self.dataset)-1)]
        
        img =sample["img"]
        assert(get_image_channel(img)==img_c)
        saliency =None
        if "saliency" in sample.keys() :saliency=sample["saliency"]
        
        if saliency is None:
            saliency=Saliency(np.ones((img.size[1],img.size[0]),dtype=np.uint8))
        else:
            saliency=Saliency(saliency)
        
        if self.saliency_method==SalienMethod.NONE:
            roi_bbox=Bbox(0,0,*fill_size,ImageSize(img))
            roi_bbox.random_move()
            return img.crop(roi_bbox.get())
        elif self.saliency_method==SalienMethod.SALIENCY_CONSTRAINT:
            anchor_point=saliency.getRandomPoint()

            roi_bbox=Bbox(0,0,*fill_size,ImageSize(img))
            roi_bbox.move_by_center(anchor_point)
            return img.crop(roi_bbox.get())
        
        else:
            return NotImplemented
    