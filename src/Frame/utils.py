

import pandas as pd
import json
import csv
from collections import OrderedDict
from functools import partial
import cv2
import os 

import numpy as np
import torch

from   torchvision import transforms



class Mapper(object):
    def __init__(self,func):
        self.func=func
    def __call__(self,objects):
        
        if isinstance(objects,tuple):
            return tuple( self.func(obj) for obj in objects )
        elif isinstance(objects, list):
            return [ self.func(obj) for obj in objects]
        elif isinstance(objects,dict):
            return { k:self.func(v) for k ,v in objects.items()}
        else:
            return self.func(objects)
        
        

        
class InverseNormalize(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    
    
to_cpu=Mapper(lambda x: x.detach().cpu()) 
    
class ToNumpy(object):
    
    """
    map a tensor ranging [0,1]  to a uin8 numpy array ranging [ 0,255]
    
    """
    def __init__(self,multiplier=255,dtype=np.uint8,transpose=False,squeeze=False):
        
        self.multiplier=multiplier
        self.dtype=dtype
        self.transpose=transpose
        self.squeeze=squeeze
    def run(self, tensor:torch.tensor):
        
        if self.transpose:
            assert(len(tensor.shape)==4),"tensor dims must be 4  when setting transpose as True !!! "
        tensor=tensor*self.multiplier
        if tensor.is_cuda: tensor=tensor.cpu()
        array=tensor.numpy().astype(self.dtype)
        if self.transpose: array=array.transpose(0,2,3,1)
        if self.squeeze: array=array.squeeze()
        return array
    def __call__(self, data):
        return Mapper(self.run)(data)
    
import os
class Folder(object):
    
    
    def __init__(self, root):
        
        self.root=root

    def exists(self,filename):
        
        return os.path.exists(os.path.join(self.folder,filename))
    
    
    def find_file(self,filename,recursion=False):
        pass
    
    def find_files_by_suffix(self,suffixes,recursion=False):
        
        def condition_func(filename,suffix):
            return filename.endswith(suffix)
        
        if not  isinstance(suffixes,(list,tuple)):
            suffixes=[suffixes]
        res=[]
        for suffix in suffixes:
            condition=partial(condition_func,suffix=suffix)
            res+=Folder.list_folder(self.root,True,condition,recursion)  
        return res
    
    
    def find_child_folders(self,condition=None):
        
        dirs=[ {"root":root,"dirs":dirs,"files":files }  for root, dirs, files in os.walk(self.root)][0]["dirs"]
        if condition is not None:
            dirs=[ d for d in dirs if condition(d)]
        dirs= [ os.path.join(self.root,d) for d in dirs]
        return dirs
        
    
    
    @staticmethod
    def list_folder(root, use_absPath=True, func=None, recursive=True):
        """
        :param root:  文件夹根目录
        :param func:  定义一个函数，过滤文件
        :param use_absPath:  是否返回绝对路径， false ：返回相对于root的路径
        :return:
        """
        root = os.path.abspath(root)
        if os.path.exists(root):
            print("遍历文件夹【{}】......".format(root))
        else:
            raise Exception("{} is not existing!".format(root))
        files = []
        # 遍历根目录,
        for cul_dir, _, fnames in sorted(os.walk(root)):

            for fname in sorted(fnames):
                path = os.path.join(cul_dir, fname)  # .replace('\\', '/')
                if func is not None and not func(path):
                    continue
                if use_absPath:
                    files.append(path)
                else:
                    files.append(os.path.relpath(path, root))
            if not recursive: break
        print("    find {} file under {}".format(len(files), root))
        return files

    
    

    
    
    
class CsvLogger(object):
    
    
    def __init__(self,root,csv_name):
        suffix=".csv"
        self.root=root
        self.csv_name=csv_name if csv_name.endswith(suffix) else csv_name+suffix
        self.csv_path=os.path.join(self.root,self.csv_name)
        
        if not  os.path.exists(self.root):
            os.makedirs(self.root)
        self.header=None
        
        if os.path.exists(os.path.join(self.root,self.csv_name)):
            self.header,_=self._read_csv(self.csv_path)
            # print("found a existing csv file and load the header: {}...".format(self.header))
        # if not Folder(self.root).exists(csv_name):
        
    def exists(self):
        return   os.path.exists(self.csv_path)
    def set_header(self,header:list):
        
        assert(self.header==None)
        self.header=header    
        self.append_one_row({ k:k for k in self.header})
        return self
    
    def get_rows(self,item=None):
        _,rows=self._read_csv(self.csv_path)
        if item is  None:
            return rows
        else:
            filtered_rows=[ row for  row  in rows if row[item[0]]==item[1]]
            return filtered_rows
    
    # def _read_csv(self,csv_path):
    #     with open(csv_path, newline='') as csvfile:
    #         spamreader = [  row for row  in csv.reader(csvfile, delimiter=',', quotechar='|') ]
    #         header=spamreader[0]
    #         rows=[]
    #         for row_val in spamreader[1:]:
    #             rows.append({ key:val for key,val in zip(header,row_val)})
    #         return header,rows
    #

    def _read_csv(self,csv_path):
        import pandas as df
        df_data = df.read_csv(csv_path)
        header= df_data.columns.to_list()
        rows=[ row.to_dict() for  idx, row in df_data.iterrows()]
        return header,rows
# def _save_csv_file(self,df):
        
    def append_one_row(self,row:dict,strict=True):
        if strict:
            assert(len(row)==len(self.header))
            assert( all(      [ (k in self.header) for k,v in row.items()])),row
            row= [  row[k] for k in  self.header]
            with open(self.csv_path, 'a+', newline='') as csvfile:
                csvw = csv.writer(csvfile)
                csvw.writerow(row)
        else:
            raise NotImplementedError
            #         try:
            #         except PermissionError:  
            #         return

    
    

import time
def get_current_time_point():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())




def get_class_name(x):
    return x.__class__.__name__



"""
get the folder name in which  the file is located
获得输入文件名，所在文件夹的名字

"""
def get_folder_name_of_file(file):
    dir_name=os.path.split(os.path.realpath(file))[0]
    return os.path.basename(dir_name)


"""

 a experiment save_root:
 
 save_root + "project_name" +"exp_name"+"run_name"

"""


class Experiment(object):
    
    def __init__(self,save_root,project_name,exp_name,run_name=None):
        
        
        
        self.save_root=save_root
        self.exp_name=exp_name
        self.run_name=run_name
        if self.run_name is None: self.run_name="runtime_"+get_current_time_point()
        exp_run_name="_".join([self.exp_name,self.run_name])
        self.save_dir=os.path.join(save_root,project_name,exp_run_name)

    def get_save_dir(self):
        
        return self.save_dir
    
import  collections
from itertools import repeat
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")
