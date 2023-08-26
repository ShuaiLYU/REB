    
import torch 

import torch.nn as nn
import os

from .base import HookBase


from .utils import CsvLogger

class CheckPoint(object):
    
    suffix=".pth"
    def __init__(self,root,model_name, extra_keys:list=[]):
        
        self.root =root
        self.model_name=model_name
        self.extra_keys=extra_keys
     
        self.csv=CsvLogger(self.root,model_name)
        if self.csv.header is not None:
            assert(self.csv.header==self.csv_head)
        else:
            self.csv.set_header(self.csv_head)
        self.model=None
    
    def bind_model(self,model:nn.Module):
        self.model=model
    
    @property
    def csv_head(self):
        return self.extra_keys+["weight_name"]
                   
    def save_a_weight(self,epoch,extra_info):
        
        #save model
        weight_name=self.model_name+"_"+"epo"+str(epoch).zfill(4)+CheckPoint.suffix
        extra_info["weight_name"]=weight_name
        if len(self.csv.get_rows())>0:
            last_weight_name=self.csv.get_rows()[-1]["weight_name"]
            if(weight_name==last_weight_name): return
        self.csv.append_one_row(extra_info)
        self._save(os.path.join(self.root,weight_name))
        
    def load_the_best(self,by_key=None):
        raise NotImplementedError
    
    def load_the_last(self):
        last_one=self.csv.get_rows()[-1]
        weight_name=last_one["weight_name"]
        weight_path=os.path.join(self.root,weight_name)
        self._load(weight_path)
        pass
    
    def load_by_name(self,weight_name):
        weight_path=os.path.join(self.root,weight_name)
        self._load(weight_path)
    
    def _save(self,weight_path):
        torch.save(self.model.state_dict(),weight_path)
        
    def _load(self,path):
        pthfile = torch.load(path)
        self.model.load_state_dict(pthfile)
        print("load weights from {}".format(path))
        
        
    def empty(self):
        files= [ file for file in os.listdir(self.root) if file.endswith(self.suffix)]
        return ( len(files)==0)

class CheckPointHook(CheckPoint,HookBase):
    
    def __init__(self,save_period,model_name, extra_keys:list):
        self.model_name=model_name
        self.extra_keys=extra_keys
        self.save_period=save_period
        
    def before_train(self):
        
        root=self.trainer.saver.save_dir
        CheckPoint.__init__(self,os.path.join(root,"checkpoint"),self.model_name, self.extra_keys)
        self.bind_model(self.trainer.model)
        
    def after_epoch(self):
        epo=self.trainer.epo
        if epo%self.save_period!=0:
            return 
        extra_info={ k:self.trainer.epo_variables[k]  for k in self.extra_keys}
        self.save_a_weight(epo,extra_info)
        
    def after_train(self):
        epo=self.trainer.epo
        extra_info={ k:self.trainer.epo_variables[k]  for k in self.extra_keys}
        self.save_a_weight(epo,extra_info)
    
    
    

class PreloadCheckPointHook(CheckPoint,HookBase):
    def __init__(self,root,model_name):
        self.root=root
        self.model_name=model_name
        
        
    def before_train(self):
        root=self.root
        CheckPoint.__init__(self,os.path.join(root,"checkpoint"),self.model_name, [])
        self.bind_model(self.trainer.model)
        self.load_the_last()
        
    