


import os
import json

import  csv
import  pandas  as pd

import numpy as np

import PIL.Image as Image
import cv2




def list_folder(root,use_absPath=True, func=None):
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
            path = os.path.join(cul_dir, fname)#.replace('\\', '/')
            if  func is not None and not func(path):
                continue
            if use_absPath:
                files.append(path)
            else:
                files.append(os.path.relpath(path,root))
    print("    find {} file under {}".format(len(files), root))
    return files


class FolderDataset(object):


    def __init__(self, root,suffixes:list=None):

        self.samples=[]
        def check_func(filename):
            if not suffixes:
                return True
            for suffix in suffixes:
                if filename.endswith(suffix):
                    return True
            return  False
        folders = [folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]
        for foldername in folders:
            subfolder = os.path.join(root, foldername)
            # print(cul_dir)
            filenames=list_folder(subfolder,True,check_func)
            for filename in filenames:
                label=foldername
                self.samples.append({"label": label, "filepath": filename})

    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        return self.samples[idx]

