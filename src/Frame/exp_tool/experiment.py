import logging
import os
from .param import Param
from .utils import get_current_time_point
__all__ = ["Experiment","get_existing_exp"]
    
    
import threading

"""
https://www.cnblogs.com/huchong/p/8244279.html
"""
class SingletonType(type):
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance


import random 
import numpy as np
import torch
def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        
        
    

class Experiment(metaclass=SingletonType):
    def __init__(self, save_root=None, project_name=None, exp_name=None, run_name=None,seed=None):
        self.save_root = save_root
        self.exp_name = exp_name
        self.run_name = run_name
        if self.run_name is None: self.run_name = "runtime_" + get_current_time_point()
        exp_run_name = "_".join([self.exp_name, self.run_name])
        self.save_dir = os.path.join(save_root, project_name, exp_run_name)
        self._meta_data={}
        # #创建单例，log模块
        # self.init_global_logger(os.path.join(self.save_dir,"log.txt"))
        
        if seed is not None:
            fix_seeds(seed)
        
        self._meta_data["seed"]=seed
    def set_attribute(self,key,val):
        self._meta_data[key]=val
        
    def get(self,key):
        return self._meta_data[key]
        
    def set_param(self,param:Param):
        self._param=param
        
    def get_param(self)->Param:
        return self._param
    
    def get_save_dir(self):
        return self.save_dir

    def set_logger(self,filename="log"):

        if (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        LOG_FORMAT = "[%(asctime)s %(name)s %(levelname)s %(pathname)s]\n %(message)s "  # 配置输出日志格式
        DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a '  # 配置输出时间的格式，注意月份和天数不要搞乱了
        logging.basicConfig(level=logging.INFO,
                            format=LOG_FORMAT,
                            datefmt=DATE_FORMAT,
                            filename=os.path.join(self.save_dir,filename)  # 有了filename参数就不会直接输出显示到控制台，而是直接写入文件
                            )
        
        
    def get_logger(self):
        return logging
    
    def info(self,string):
        logging.info(string)
    
    

# def get_existing_exp():
#     # try:
#     EXPER = Experiment()
#     print("EXPER is  existing")
#     return EXPER

def get_existing_exp():
    try:
        EXPER = Experiment()
        # print("EXPER is  existing")
        return EXPER
    except:
        print("EXPER is not existing !!!")
        return None
    
    