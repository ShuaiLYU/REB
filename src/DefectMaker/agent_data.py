from tqdm import tqdm
from PIL import Image
import random
import os
import numpy as np
import torch
class DmAgencyWrapper(object):
    
    
    def __init__(self,dataset,transform=None,**kwargs):
        self.dataset=dataset
        self.data_type =self.dataset.defect_name
        

        self.transform_after=transform

#         self.cp_collate_fn=get_cp_collate_fn(["img","cp_img","cps_img"])
        
        self.unpair=kwargs.get("unpair",False)
        self.return_mask=kwargs.get("return_mask",False)

        self.dm_names={
        "imp_cp":"/media/lyushuai/Data/metec_proxy",
        "imp_cps":"/media/lyushuai/Data/metec_proxy",
        "perline_noise":"/media/lyushuai/Data/metec_proxy2"
        }
        self.load_dataset()
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        idx=idx%len(self.dataset)
        
        sample=self.dm_run(self.dataset[idx])
        if self.transform is not None:
            sample={ key: self.transform(val) if "img" in key else val for key,val in sample.items() } 
        
        # print(sample.keys(),222)
        items= self.filter_by_keys(sample)
        return items 
    
    
    def load_dataset(self):
        #      return {"img":img,"label":label,"label_name":label_name,"imagepath":imagepath,"saliency":saliency,"mask":mask}
        


        def get_child_folders(dirname):

            return next(os.walk(dirname))[1]
    
        synth_data=[] 
        
#         print(len(self.dataset))
#         return
        
        for idx,sam in tqdm(enumerate(self.dataset)):
            synth_sample={}
            # print(sam["imagepath"])
            img_id=int(os.path.basename(sam["imagepath"]).split(".")[0])
            img_id_str="{:0>5d}".format(img_id)
            img_id_str="{:0>5d}".format(idx)
            # print(img_id_str)
            for dm_name,dm_path in self.dm_names.items():
                next_path=os.path.join(dm_path,self.data_type,img_id_str)
                child_folders=[  os.path.join(next_path,folder)  for folder in get_child_folders(next_path) if dm_name+"_" in folder ]
                synth_sample[dm_name]=child_folders
            # break
            # for dm_name, paths in synth_sample.items():
            #     print(img_id,dm_name,len(paths))
            synth_data.append(synth_sample)
            # print(idx,synth_data[-1].keys())
        self.synth_data=synth_data
        print("find files finished")
            
            
    def __len__(self):
        return  len(self.dataset)
        
    def __getitem__(self, idx):
        

                                                                        
        def read_img(img_path):
            if img_path is None:
                return None
            else:
                return Image.open(img_path)
            
        assert ( isinstance(self.dataset[idx],dict))    
        sample=self.dataset[idx]

        # sample["img"]=read_img(self.dataset[idx]["img"])
        # sample["saliency"]=read_img(self.dataset[idx]["saliency"])
        # print(sample["imagepath"])
        synth_idx=idx
        if self.unpair:     
            synth_idx=random.randint(0,len(self)-1)
        synth_info=self.synth_data[idx]
        def get_synth_sam(synth_info):
            synth_data={}
            for k, paths in synth_info.items():
                path=random.choice(paths)
                # print(path)
                synth_data["img_"+k]=read_img(os.path.join(path,"img.jpg"))
                if self.return_mask:
                    synth_data["mask_"+k]=read_img(os.path.join(path,"mask.png"))
            return synth_data
        synth_data=get_synth_sam(synth_info)
        for k,v in synth_data.items():
            sample[k]=v
                                               
        if  self.transform_after is not None:
            sample= {k:self.transform_after(v) if "img" in k else v for k, v in sample.items() }
        return  sample
    
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
            # print(ys)
            data_batch={"img":xs,"class":ys}
            if len(masks)>0: data_batch["mask"]=masks
            if len(reconsts)>0: data_batch["reconsts"]=reconsts                                
            def format_data(fmt,data):
                res=[]
                for form in fmt:
                    if isinstance(form,str):
                        res.append(data[form])
                    else:
                        res.append(format_data(form,data))
                return res
            batch=format_data(data_format,data_batch)
            # for bat in batch: print(bat.shape)
            return batch
        return collate_func
        
def get_dm_agent_wrapper(*args,**kwargs):
    return DmAgencyWrapper