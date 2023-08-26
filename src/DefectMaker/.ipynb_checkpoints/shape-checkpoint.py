
from PIL import Image
import random
import numpy as np
import functools
import cv2 as cv
from .base import *
from .perlin_noise_mask import gen_perline_noise_mask
from .bezier_mask import BezierMaskGenerator,gen_bbezier_mask

from .perlin_noise_mask import PerlinNoise
class PerlineShapeMaker(DmDataBase):
    
    
    def __init__(self,perlin_scale_range=None,threshold_set=None, **kwargs):

        if perlin_scale_range is None: perlin_scale_range = [0,3]
        perlin_scale_x_set=set([2**scale for scale in list(np.arange (*perlin_scale_range,1))])
        perlin_scale_y_set=set([2**scale for scale in list(np.arange (*perlin_scale_range,1))])
        if threshold_set is  None : threshold_set=set(np.arange (*[0.5,0.8],0.1))
        revise_set=set([False])
        func_combs=[]
        for perlin_scale_x in perlin_scale_x_set:
            for perlin_scale_y in perlin_scale_y_set:
                for threshold in threshold_set:
                    for revise in revise_set:
                        param_comb={"perlin_scale_xy":(int(perlin_scale_x),int(perlin_scale_y)),"revise":revise,"threshold":threshold} 
                        func= functools.partial(self.gen_perline_noise_mask, **param_comb) 
                        func_combs.append(func)
                    
        self.func_combs =func_combs
    
    @staticmethod
    def gen_perline_noise_mask(img_size,perlin_scale_xy,threshold=0.5,revise=False):
        # print(img_size,perlin_scale_xy,threshold,revise)
        
        return gen_perline_noise_mask(img_size,perlin_scale_xy,threshold,revise)
    
    def get_one_shape(self,img_size):
        idx=random.randint(0,len(self)-1)
        shape_array=self.get_one(idx,img_size=img_size)
        return Shape(shape_array,unpad=False)
    



        
class BezierShapeMaker(DmDataBase):
    
    
    def __init__(self,bezier_point_num_range=[5,20],k_range=[0.1,0.3], **kwargs):
        
    
        bezier_point_num_set=set(range(*bezier_point_num_range,1))
        k_set=set(list(np.arange(*k_range,0.1)))
        
        func_combs=[]
        for bezier_point_num in bezier_point_num_set:
            for k in k_set:
                # param_comb={"img_size":img_size,"img_c":img_c,"mean":mean,"fluct":fluct,"scale":scale} 
                param_comb={"bezier_point_num":bezier_point_num,"k":k} 
                func= functools.partial(self.gen_bbezier_mask, **param_comb) 
                func_combs.append(func)
        self.func_combs =func_combs
        
    
    def get_one_shape(self,img_size):
        idx=random.randint(0,len(self)-1)
        shape_array=self.get_one(idx,img_size=img_size)
        return Shape(shape_array,unpad=False)
    
    @staticmethod
    def gen_bbezier_mask(img_size,bezier_point_num,k):
        return gen_bbezier_mask(img_size,bezier_point_num,k)
    
class RectShapeMaker(object):

    def __init__(self,img_size,area_ratio_range, aspect_ratio):
        self.area_ratio_range=area_ratio_range
        self.aspect_ratio=aspect_ratio
        self.img_size=img_size
        param_comb={"img_size":img_size,"area_ratio_range":area_ratio_range,"aspect_ratio":aspect_ratio} 
        self.func=functools.partial(self.get_rect_param, **param_comb) 
 
        
    @staticmethod
    def get_rect_param(img_size,area_ratio_range, aspect_ratio,restrict_bbox=None):
        w, h =  img_size[0], img_size[1]
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(area_ratio_range[0], area_ratio_range[1]) * w * h
        
        
        
        # sample in log space
        log_aspect = [math.log(aspect_ratio), math.log(1 / aspect_ratio)]
        log_aspect = random.uniform(min(log_aspect), max(log_aspect))
        aspect = math.exp(log_aspect)

        rect_w = int(round(math.sqrt(ratio_area * aspect)))
        rect_h = int(round(math.sqrt(ratio_area / aspect)))
        if(rect_w>img_size[0]):
            rect_w=img_size[0]
            rect_h=int(round(ratio_area/rect_w))
        if(rect_h>img_size[1]):
            rect_h=img_size[1]
            rect_w=int(round(ratio_area/rect_h))
        
        return  rect_w,rect_h

    def get_one_shape(self,*args,**kwargs):
        rect_w,rect_h=self.func()
        return RectShape(rect_w,rect_h)
    
class RectScarShapeMaker(object):

    def __init__(self,width_range=[2,16], height_range=[10,25], rotation_range=[-45,45]):
        self.width_range=width_range
        self.height_range=height_range
        self.rotation_range=rotation_range
        param_comb={"width_range":width_range,"height_range":height_range,"rotation_range":rotation_range} 
        self.func=functools.partial(self.get_rect_param, **param_comb) 
        
    @staticmethod
    def get_rect_param(width_range,height_range, rotation_range):
        width=random.randint(*width_range)
        height=random.randint(*height_range)
        rotation=random.randint(*rotation_range)
        return  width,height,rotation
    def get_one_shape(self):
        width,height,rotation=self.func()
        return RectShape(width,height,rotation)
    
    

    
    
class BezierRectShapeMaker(BezierShapeMaker):

    def __init__(self , img_size,area_ratio_range, aspect_ratio,bezier_point_num_range=[5,10],k_range=[0.2,0.5]):
        
        self.area_ratio_range=area_ratio_range
        self.aspect_ratio=aspect_ratio
        self.img_size=img_size
        super(BezierRectShapeMaker,self).__init__(bezier_point_num_range,k_range)

    def get_one_shape(self):
        rect_w,rect_h=RectShapeMaker.get_rect_param(self.img_size,self.area_ratio_range,self.aspect_ratio)
        return super(BezierRectShapeMaker,self).get_one_shape((rect_w,rect_h))
    
    
class BezierRectScarShapeMaker(BezierShapeMaker):

    def __init__(self,width_range=[2,16], height_range=[10,25], rotation_range=[-45,45],bezier_point_num_range=[5,20],k_range=[0.1,0.3]):
        self.width_range=width_range
        self.height_range=height_range
        self.rotation_range=rotation_range
        super(BezierRectScarShapeMaker,self).__init__(bezier_point_num_range,k_range)

    def get_one_shape(self):
        width,height,rotation=RectScarShapeMaker.get_rect_param(self.width_range,self.height_range,self.rotation_range)
        shape= super(BezierRectScarShapeMaker,self).get_one_shape((width,height))
        return CurveShape(shape.array,rotation)
    


"""
2023年2月21日15:10:06 

"""
    
    
class RectClumpShapeMaker(RectShapeMaker):

    def __init__(self,img_size,area_ratio_range, aspect_ratio,shape_scale_range=[1,5],shape_aspect_ratio=2):
        self.shape_scale_range=shape_scale_range
        self.shape_aspect_ratio=shape_aspect_ratio
        RectShapeMaker.__init__(self,img_size,area_ratio_range,aspect_ratio)
        
    
    
    def gen_random_noise_img(self,img_size,img_c=1,mean=127,fluct=10,scale=1,aspect_ratio=1):
        assert(len(img_size)==2)

        low = max(mean - fluct ,0)
        high = min(mean + fluct+1  ,255)
        
        scale_img_size= ( max(int(img_size[0]/scale*aspect_ratio),1),max(int(img_size[0]/scale/aspect_ratio),1))
        shape=(scale_img_size[1],scale_img_size[0]) if img_c==1 else (scale_img_size[1],scale_img_size[0],img_c)
        defect = np.random.randint(low, high, shape)
        return Image.fromarray(defect.astype(np.uint8)).resize(img_size)
    
    def get_one_shape(self,shape_size=None):
        if shape_size==None:
            shape_size=self.func()
        
        scale=random.uniform(*self.shape_scale_range)
        log_aspect = [math.log(self.shape_aspect_ratio), math.log(1 / self.shape_aspect_ratio)]
        log_aspect = random.uniform(min(log_aspect), max(log_aspect))
        aspect = math.exp(log_aspect)
        array=self.gen_random_noise_img(shape_size,1,127,50,scale,aspect)
        array=np.where(np.array(array)>128+10,255,0)
        return Shape(array,unpad=False)
    
class BezierClumpShapeMaker(BezierShapeMaker,RectClumpShapeMaker):

    def __init__(self, img_size,area_ratio_range, aspect_ratio,shape_scale_range=[1,5],shape_aspect_ratio=2,bezier_point_num_range=[5,10],k_range=[0.2,0.5]):
        RectClumpShapeMaker.__init__(self,img_size,area_ratio_range, aspect_ratio,shape_scale_range,shape_aspect_ratio)
        BezierShapeMaker.__init__(self,bezier_point_num_range,k_range)
        

    
    def get_one_shape(self,shape_size=None):
        shape1=RectClumpShapeMaker.get_one_shape(self,shape_size)
        if shape_size==None:
            shape_size=self.func()
        shape2=BezierShapeMaker.get_one_shape(self,shape1.get_wh())
        return shape1.intersection(shape2)
        


    
class PerlineRectShapeMaker(PerlineShapeMaker):

    def __init__(self,img_size,area_ratio_range, aspect_ratio,perlin_scale_range=[0,3]):
        self.area_ratio_range=area_ratio_range
        self.aspect_ratio=aspect_ratio
        self.img_size=img_size
        super(PerlineRectShapeMaker,self).__init__(perlin_scale_range)
    
    def get_one_shape(self):
        rect_w,rect_h=RectShapeMaker.get_rect_param(self.img_size,self.area_ratio_range,self.aspect_ratio)
        return super(PerlineRectShapeMaker,self).get_one_shape((rect_w,rect_h))
    
    
    
    
class PerlineRectScarShapeMaker(PerlineShapeMaker):

    def __init__(self,width_range=[2,16], height_range=[10,25], rotation_range=[-45,45],perlin_scale_range=[0,3]):
        self.width_range=width_range
        self.height_range=height_range
        self.rotation_range=rotation_range
        super(PerlineRectScarShapeMaker,self).__init__(perlin_scale_range)

    def get_one_shape(self):
        width,height,rotation=RectScarShapeMaker.get_rect_param(self.width_range,self.height_range,self.rotation_range)
        
        shape=super(PerlineRectScarShapeMaker,self).get_one_shape((width,height))
        return CurveShape(shape.array,rotation)
    

    
    
    
def shape_smooth_edge(shape,kernel_size=5):
    mask=np.array(shape,dtype=np.uint8)*255
    kernel_size=int(kernel_size//2*2+1)
    # print(kernel_size)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_erode = cv.morphologyEx(mask, cv.MORPH_ERODE, element)
    mask_erode_gauss = cv.GaussianBlur(mask_erode, (kernel_size, kernel_size), kernel_size)
    return np.array(mask_erode_gauss,dtype=np.float16)/255