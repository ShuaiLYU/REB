from PIL import Image
import random
import numpy as np
import functools
import torch
from torchvision import transforms
from .base import *
from .perlin_noise_mask import gen_perline_noise_mask
from .perlin_noise_mask import PerlinNoise
from .bezier_mask import BezierMaskGenerator,gen_bbezier_mask
from .shape import *
from .fill import *
from .agent_data import get_dm_agent_wrapper
import cv2

import random
import numpy as np
class HypterParam(object):
    
    def __init__(self,value_range:(tuple,int,float)):
        
        self.value_range=value_range
        if  isinstance(value_range,tuple):
            assert(self.value_range[1]>=self.value_range[0])
      
    def get(self):
        
        if isinstance(self.value_range, tuple):
            bias= (random.random())*(self.value_range[1]-self.value_range[0])
            if  len(self.value_range)>2: bias= np.around(bias/self.value_range[2])*self.value_range[2]
            return  self.value_range[0]+bias
        elif isinstance(self.value_range, list):
            return random.choice(self.value_range)
        else: 
            return self.value_range

class DefectMaker(object):
    
    def __init__(self, fill_data,shape_data,fuse_weight_range=(0.5,1),saliency_method:SalienMethod=SalienMethod.IMAGE_FIT,
                 blur_ksize=5,blur_prob=0,**kwargs):
        self.fill_data=fill_data
        self.shape_data=shape_data
        self.saliency_method=saliency_method

        self.fuse_weight_set=HypterParam(fuse_weight_range)

        self.blur_prob=blur_prob
        self.blur_ksize=blur_ksize
        self.return_shape=kwargs.get("return_shape",False)
        self.return_fill=kwargs.get("return_fill",False)
        self.return_mask=kwargs.get("return_mask",False)
        
        self.adversarial=kwargs.get("adversarial",False)
        colorJitter=0.1
        self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)
        
        self.poission_fuse=kwargs.get("poission_fuse",False)

    def make_defect(self, sample:dict):
        
        
        attempt = 0
        max_attempts=4
        while attempt < max_attempts: # 最多尝试次数
            try:
                dm_sample=self._make_defect(sample) # 举例，执行一个函数
                break
            except:
                # 如果代码执行失败，这里会捕获到异常
                # 程序会跳转到这里并重新执行 try 代码块
                attempt += 1
                # print(f"make_defect 执行失败，尝试重新执行。尝试次数：{attempt}")
        else:
            # 如果尝试执行多次后仍然失败，这里会被执行
            raise Exception(f"执行失败次数超过了 {max_attempts} 次，程序终止。")
        return dm_sample
        
    def _make_defect(self, sample:dict):
        img =sample["img"]
        img_size=ImageSize(img)
        saliency =sample["saliency"]
        if saliency is None:
            saliency=np.ones(shape=(img_size.h,img_size.w),dtype=np.uint8)
        if self.saliency_method==SalienMethod.IMAGE_FIT:
            shape_size=ImageSize(img).get()
            shape=self.shape_data.get_one_shape(shape_size)
            fill=self.fill_data.get_one_fill(shape_size,get_image_channel(img))
            roi_bbox=Bbox(0,0,*shape_size,img_size)
        elif self.saliency_method==SalienMethod.NONE:
            #shape
            shape=self.shape_data.get_one_shape()
            shape_size=ImageSize(shape.array).get()
            # random location        
       
            roi_bbox=Bbox(0,0,*shape_size,img_size)
            roi_bbox.random_move()
            #fill
            fill=self.fill_data.get_one_fill(shape_size,get_image_channel(img))
   
        elif self.saliency_method==SalienMethod.SALIENCY_FIT:
            saliency=Saliency(saliency)
            shape_size=ImageSize(Shape(saliency).get()).get()
            roi_bbox=saliency.to_bbox()
            
            shape=self.shape_data.get_one_shape(shape_size)
            fill=self.fill_data.get_one_fill(shape_size,get_image_channel(img))

            
        elif self.saliency_method==SalienMethod.SALIENCY_FIT_INTER:
            saliency=Saliency(saliency)
            shape_size=ImageSize(Shape(saliency).get()).get()
            roi_bbox=saliency.to_bbox()
            
            shape=self.shape_data.get_one_shape(shape_size)
            shape=shape.intersection(saliency.crop(roi_bbox))
    
            fill=self.fill_data.get_one_fill(shape_size,get_image_channel(img))
    
            
        
        elif self.saliency_method==SalienMethod.SALIENCY_CONSTRAINT:
            saliency=Saliency(saliency)
            anchor_point=saliency.getRandomPoint()
            
            shape=self.shape_data.get_one_shape()
            shape_size=ImageSize(shape.array).get()

            roi_bbox=Bbox(0,0,*shape_size,img_size)
            roi_bbox.move_by_center(anchor_point)
            fill=self.fill_data.get_one_fill(shape_size,get_image_channel(img))

        
        elif self.saliency_method==SalienMethod.SALIENCY_CONSTRAINT_INTER:
            
            saliency=Saliency(saliency)
            anchor_point=saliency.getRandomPoint()            # get location by saliency
            
            shape=self.shape_data.get_one_shape()
            shape_size=ImageSize(shape.array).get()

        
            # print(shape_size)
            roi_bbox=Bbox(0,0,*shape_size,img_size)
            roi_bbox.move_by_center(anchor_point)
            
            #intersection
            shape=shape.intersection(saliency.crop(roi_bbox))
            
            fill=self.fill_data.get_one_fill(shape_size,get_image_channel(img))

        img_defect=self.run_fuse(img,fill,shape,roi_bbox,self.fuse_weight_set.get())
        dm_sample={"img":img_defect}
        if self.return_shape: dm_sample["shape"]=shape.get()
        if self.return_fill: dm_sample["fill"]=fill
        if self.return_mask:dm_sample["mask"]=shape.to_mask(bbox=roi_bbox).get()
        if self.adversarial: dm_sample["img_aug"]=self.run_fuse(img,fill,shape,roi_bbox,random.random()*0.01)
        return dm_sample
        

    # def get_pad_fill(self,fill:Image,bbox:Bbox):
#         img_c=get_image_channel(fill)
#         img_size=bbox.img_size.get()
    
#         if img_c>1:
#             array = np.zeros((img_size[1], img_size[0],3), np.uint8)
#         else:
#             array = np.zeros((img_size[1], img_size[0]), np.uint8)
#         array[bbox.y1:bbox.y2,bbox.x1:bbox.x2]=np.array(fill)      
#         return Image.fromarray(array)
    
    def run_fuse(self,img:Image, fill:Image,shape:Shape,bbox:Bbox,weight=1):
        
        # print("weight ： ",weight)
        if self.colorJitter is not None:
            fill=self.colorJitter(fill)
        use_poission=True
        if self.poission_fuse:
            mask=shape.get().astype(np.uint8)*255

            loc=((bbox.x1+bbox.x2)//2,(bbox.y1+bbox.y2)//2)
            dst_img = cv2.seamlessClone(np.array(fill),np.array(img),  mask, loc, cv2.NORMAL_CLONE)
            
        else:
            img=np.array(img,dtype=np.float16)
            fill=np.array(fill,dtype=np.float16)

            dst_img=img

            if random.random()<self.blur_prob:
                mask =shape_smooth_edge(shape.get(),self.blur_ksize)*weight
            else:
                mask =shape.get().astype(np.float16)*weight
            # print(np.unique(mask))
            if len(img.shape)==3:
                mask = np.expand_dims(mask, axis=-1)

            dst_img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]=mask*fill+dst_img[bbox.y1:bbox.y2, bbox.x1:bbox.x2]*(1-mask)

       
        
        return Image.fromarray(dst_img.astype(np.uint8))
        


        
########################################################################################################################     
    



######################################################################################################################################################



class DmsCombineMethod(Enum):
    ALL=0
    RANDOM_ONE=0
    
    
    
class DmDatasetWrapper(object):
    
    def __init__(self,dataset,dms:dict, keys=None,new_length=None,transform=None,
                 combine_method:DmsCombineMethod=DmsCombineMethod.ALL,**kwargs):
        
        self.dataset=dataset
        self.dms=dms
        self.keys=keys
        self.new_length=new_length
        self.transform=transform
        self.combine_method=combine_method
        if self.new_length is not None:
            self._iidxes=[ i%len(self.dataset) for i  in range(self.new_length)]
        else:
            self._iidxes=[ i%len(self.dataset) for i  in range(len(self.dataset))]
            
    def __len__(self):
        return len(self._iidxes)
    
    def __getitem__(self, idx):
        idx=self._iidxes[idx]
        sample=self.dm_run(self.dataset[idx])
        if self.transform is not None:
            sample={ key: self.transform(val) if "img" in key else val for key,val in sample.items() } 
        
        # print(sample.keys(),222)
        items= self.filter_by_keys(sample)
        return items 
    
    def dm_run(self,sample:dict) ->dict:
        assert("img" in sample.keys())
        assert("saliency" in sample.keys())
        if self.combine_method ==DmsCombineMethod.ALL:
            for dm_name, dm in self.dms.items():
                dft_sample=dm.make_defect(sample)
                for k, v in dft_sample.items():
                    kk=k+"_"+dm_name
                    assert(kk not in sample.keys())
                    sample[kk]=v
        elif self.combine_method==DmsCombineMethod.RANDOM_ONE:
            dm_name="dm"
            dm=random.choice(list(self.dms.values()))
            dft_sample=dm.make_defect(sample)
            for k, v in dft_sample.items():
                kk=k+"_"+dm_name
                assert(kk not in sample.keys())
                sample[kk]=v
        else:
            return NotImplemented
        return sample

    def filter_by_keys(self,sample):
        if self.keys==None:
            items=sample
        elif  isinstance(self.keys,tuple) or isinstance(self.keys,list):
            items= [sample[k] for k in self.keys]
        else:
            items=sample[self.keys]
        return items
    
    def get_collate_func(self,key_label_map:dict,data_format:list=["img","class"],mask_size=None):
        def to_str(fmt):
            _str=""
            for idx,form in enumerate(fmt):
                if isinstance(form,str):
                    _str+=form
                else:
                    _str+=to_str(form)
            return _str
        str_data_format=to_str(data_format)
        
        
        
        def collate_func(items):
            xs,ys,masks,reconsts=[],[],[],[]
            for item in items:    
                # print(item.keys())
                for k,val in key_label_map.items():
                    item[k]=np.array(item[k])
                    xs.append(torch.tensor(item[k]) if not torch.is_tensor(item[k]) else item[k] ) 
                    ys.append(torch.tensor(val) if not torch.is_tensor(val) else val) 
                    if "mask" in str_data_format:
                        if k =="img": 
                            mask =torch.zeros(mask_size)
                        else:
                            mask=np.array(item[k.replace("img","mask")])
                        mask=(torch.tensor(mask) if not torch.is_tensor(mask) else mask)
                        masks.append(mask)    
                    if "reconst" in str_data_format:
                        reconsts.append(torch.tensor(item["img"]) if not torch.is_tensor(item["img"]) else item["img"] ) 
            xs,ys=torch.stack(xs),torch.stack(ys)
            if len(masks)>0:masks=torch.stack(masks)
            if len(reconsts)>0:
                reconsts=torch.stack(reconsts)
                reconsts=F.interpolate(reconsts,size=mask_size,mode='bilinear').permute(0,2,3,1)
            
            data_batch={"img":xs,"class":ys,"mask":masks,"reconsts":reconsts}                                
            def format_data(fmt,data):
                res=[]
                for form in fmt:
                    if isinstance(form,str):
                        res.append(data[form])
                    else:
                        res.append(format_data(form,data))
                return res  
            return format_data(data_format,data_batch)  
        return collate_func
        
    
    
def get_defect_maker(dataset,shape_name,fill_name,shape_param:dict(),fill_param:dict(),make_param:dict()):
    
    assert (shape_name in ["PerlineShapeMaker","BezierShapeMaker","RectShapeMaker","RectScarShapeMaker","RectClumpShapeMaker","BezierClumpShapeMaker"
                           ,"BezierRectShapeMaker","PerlineRectShapeMaker","BezierRectScarShapeMaker","PerlineRectScarShapeMaker"])
    assert (fill_name in ["RandomNoiseFillMaker","CutFillMaker"])
    assert ("saliency_method" in make_param.keys())
    assert ("fuse_weight_range" in make_param.keys())
    shape_data=eval(shape_name)(**shape_param)
    fill_data=eval(fill_name)(**fill_param)
    
    return DefectMaker(fill_data,shape_data,**make_param)
    
    
    

def get_data_wrapper(dataset,dms,keys=None,new_length=None,transform=None,**kwargs):
    return DmDatasetWrapper(dataset,dms,keys,new_length,transform,**kwargs)


def merge_dict(dict1,dict2):
    dict_merge={}
    for k,v in dict1.items():
        assert(k not in dict_merge.keys()),k
        dict_merge[k]=v
    for k,v in dict2.items():
        assert(k not in dict_merge.keys()),k
        dict_merge[k]=v
    return dict_merge

"""
the implementation of 
https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf
the dafault params  used in the function "get_default_cutpaste_wrapper" and "get_default_cutpastescar_wrapper"
"""



    
    
def get_cutpaste_wrapper(dataset,dm_name,shape_param:dict(),fill_param:dict(),make_param:dict()):
    shape_name="RectShapeMaker"
    fill_name="CutFillMaker"
    dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
    return functools.partial(get_data_wrapper, dms=dms) 


def get_cutpastescar_wrapper(dataset,dm_name,shape_param:dict(),fill_param:dict(),make_param:dict()):
    shape_name="RectScarShapeMaker"
    fill_name="CutFillMaker"
    dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
    return functools.partial(get_data_wrapper, dms=dms) 


def get_default_cutpaste_dm(dataset):
    dm_name="cp"
    shape_name="RectShapeMaker"
    fill_name="CutFillMaker" 
    assert(hasattr(dataset,"img_size"))
    shape_param={"img_size":dataset.img_size,"area_ratio_range":[0.02,0.15],"aspect_ratio":3.3}
    fill_param={"dataset":dataset,"saliency_method":SalienMethod.NONE}
    make_param={"fuse_weight_range":(0.99,1),"saliency_method":SalienMethod.NONE}
    
    dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
    return dms

def get_default_cutpastescar_dm(dataset):
    dm_name="cps"
    shape_name="RectScarShapeMaker"
    fill_name="CutFillMaker"
    assert(hasattr(dataset,"img_size"))
    shape_param={"width_range":[2+5,16+5],"height_range":[10+5,25+5],"rotation_range":[-45,45]}
    fill_param={"dataset":dataset,"saliency_method":SalienMethod.NONE}
    make_param={"fuse_weight_range":(0.99,1),"saliency_method":SalienMethod.NONE}
    dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
    return dms



def get_default_cutpaste_wrapper(dataset):
    
    return functools.partial(get_data_wrapper, dms=get_default_cutpaste_dm(dataset)) 

def get_default_cutpastescar_wrapper(dataset):
    return functools.partial(get_data_wrapper, dms=get_default_cutpastescar_dm(dataset)) 

def get_default_cutpaste3way_wrapper(dataset):
    dms=merge_dict(get_default_cutpaste_dm(dataset),get_default_cutpastescar_dm(dataset))
    return functools.partial(get_data_wrapper, dms=dms) 


"""
improved cutpaste algorithm methods 
"""

def get_imporved_cutpaste_dm(dataset):
    dm_name="imp_cp"
    shape_name="BezierRectShapeMaker"
    fill_name="CutFillMaker" 
    assert(hasattr(dataset,"img_size"))
    shape_param={"img_size":dataset.img_size,"area_ratio_range":[0.02,0.15],"aspect_ratio":3.3,"bezier_point_num_range":[5,7],"k_range":[0.2,0.5]}
    fill_param={"dataset":dataset,"saliency_method":SalienMethod.SALIENCY_CONSTRAINT}
    make_param={"fuse_weight_range":1,"saliency_method":SalienMethod.SALIENCY_CONSTRAINT,"blur_ksize":3,"blur_prob":0.5}
    dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
    return dms
def get_imporved_cutpastescar_dm(dataset):
    dm_name="imp_cps"
    shape_name="BezierRectScarShapeMaker"
    fill_name="CutFillMaker"
    assert(hasattr(dataset,"img_size"))
    shape_param={"width_range":[2,16],"height_range":[10,25],"rotation_range":[-45,45],"bezier_point_num_range":[5,7],"k_range":[0.2,0.5]}
    fill_param={"dataset":dataset,"saliency_method":SalienMethod.SALIENCY_CONSTRAINT}
    make_param={"fuse_weight_range":1,"saliency_method":SalienMethod.SALIENCY_CONSTRAINT,"blur_ksize":3,"blur_prob":0.5}
    dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
    return dms

def get_imporved_cutpaste_wrapper(dataset):
    return functools.partial(get_data_wrapper, dms=get_imporved_cutpaste_dm(dataset)) 
  

def get_imporved_cutpastescar_wrapper(dataset):
    return functools.partial(get_data_wrapper, dms=get_imporved_cutpaste_dm(dataset)) 

def get_imporved_cutpastes3way_wrapper(dataset):
    
    dms=merge_dict(get_imporved_cutpaste_dm(dataset),get_imporved_cutpastescar_dm(dataset))

    return functools.partial(get_data_wrapper, dms=dms) 
#####################################################################################################################################





# def get_perline_cutpaste_dm(dataset):
#     dm_name="per_cp"
#     shape_name="PerlineRectShapeMaker"
#     fill_name="CutFillMaker"
#     assert(hasattr(dataset,"img_size"))
#     shape_param={"img_size":dataset.img_size,"area_ratio_range":[0.02,0.15],"aspect_ratio":3.3,"perlin_scale_range":[0,3]}
#     fill_param={"dataset":dataset,"saliency_method":SalienMethod.NONE}
#     make_param={"fuse_weight_range":1,"saliency_method":SalienMethod.SALIENCY_CONSTRAINT,"blur_ksize":3}
#     dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
#     return dms
# def get_perline_cutpastescar_dm(dataset):
#     dm_name="per_cps"
#     shape_name="PerlineRectScarShapeMaker"
#     fill_name="CutFillMaker"
#     assert(hasattr(dataset,"img_size"))
#     # bug when rect size is too small , modify from [2,16] to [4,16]
#     shape_param={"width_range":[4+5,16+5],"height_range":[10+5,25+5],"rotation_range":[-45,45],"perlin_scale_range":[0,3]}
#     fill_param={"dataset":dataset,"saliency_method":SalienMethod.NONE}
#     make_param={"fuse_weight_range":1,"saliency_method":SalienMethod.SALIENCY_CONSTRAINT,"blur_ksize":3}
#     dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
#     return dms
#
# def get_perline_cutpaste_wrapper(dataset):
#     return functools.partial(get_data_wrapper, dms=get_perline_cutpaste_dm(dataset))
#
#
# def get_perline_cutpastescar_wrapper(dataset):
#     return functools.partial(get_data_wrapper, dms=get_perline_cutpaste_dm(dataset))
#
# def get_perline_cutpastes3way_wrapper(dataset):
#
#     dms=merge_dict(get_perline_cutpaste_dm(dataset),get_perline_cutpastescar_dm(dataset))
#     return functools.partial(get_data_wrapper, dms=dms)
#
#
#
#
# def get_perline_noise_dm(dataset):
#     dm_name="perline_noise"
#     shape_name="PerlineShapeMaker"
#     fill_name="RandomNoiseFillMaker"
#     shape_param={"perlin_scale_range":[0,4]}
#     fill_param=dict(mean_range=[50,200],mean_step=10,fluct_range=[0,50],fluect_step=5,scale_range=[0,3],aspect_ratio_range=[0,3])
#     make_param={"fuse_weight_range":(0.5,1),"saliency_method":SalienMethod.SALIENCY_FIT_INTER,"blur_ksize":3,"blur_prob":0.5}
#     dms={dm_name:get_defect_maker(dataset,shape_name,fill_name,shape_param,fill_param,make_param)}
#     return dms
#
# def get_perline_noise_wrapper(dataset):
#     return functools.partial(get_data_wrapper, dms=get_perline_noise_dm(dataset))