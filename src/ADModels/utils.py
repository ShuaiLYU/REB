import torch.nn as nn

import os
import torch
from torchvision.models import resnet18

"""


"""
class ModuleCP(nn.Module):
    
    
    def __init__(self):
        super(ModuleCP,self).__init__()
        
        self._folder=None
    
    
    def bind_folder(self,folder):
        self._folder=folder
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
            print("makedirs: {}".format(folder))
        
    
    def _save_cp(self,checkpoint,cp_name):
        cp_path=os.path.join(self._folder,cp_name)
        
        assert not os.path.exists(cp_path),\
        " save failed ! checkpoint {} has been existed in {}".format(cp_name,self._folder)
        torch.save(checkpoint,cp_path)
        
    def _load_cp(self,cp_name):
        cp_path=os.path.join(self._folder,cp_name)
        assert os.path.exists(cp_path),\
        " load failed ! checkpoint {} is not existed in {}".format(cp_name,self._folder)
        checkpoint=torch.load(cp_path)
        return checkpoint

        
    def load_state_dict(self,state_dict):
        for k, v in state_dict.items():
            if k not in self.state_dict():
                print("register buffer: {} tensor shape{}".format(k,v.shape))
                self.register_buffer(k,v)
        super(ModuleCP,self).load_state_dict(state_dict)
