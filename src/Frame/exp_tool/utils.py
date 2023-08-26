
__all__ = ["get_folder_name_of_file",
           "get_current_time_point"]

import  os
"""

 a experiment save_root:

 save_root + "project_name" +"exp_name"+"run_name"



"""



import time
def get_current_time_point():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())



"""
get the folder name in which  the file is located
获得输入文件名，所在文件夹的名字

"""
def get_folder_name_of_file(file):
    dir_name=os.path.split(os.path.realpath(file))[0]
    return os.path.basename(dir_name)





class HyperParam(object):
    def __init__(self, initVal, finalVal=-1., beginStep=-1, endStep=-1):
        self.initVal = initVal
        self.finalVal = finalVal
        self.beginStep = beginStep
        self.endStep = endStep
        assert self.beginStep<=self.endStep
    def __call__(self, step, **kwargs):
        val = self.initVal
        if self.beginStep == self.endStep:
            return val
        step = step + (self.beginStep - step) * (self.beginStep > step) + (self.endStep - step) * (self.endStep < step)
        val += self.riseCosin((step - self.beginStep) / (self.endStep - self.beginStep)) * (self.finalVal - self.initVal)
        return val

    def riseCosin(self,x):
        return (np.cos((x + 1) * np.pi) + 1) / 2



    
# class Abbreviation:
    
#     pass


# separation 
# decollator
class ProjectName(object):

    def __init__(self,param,separator=",",connector="-"):
        self.sepa=separator
        self.con=connector
        self.stri=self.to_string(param)
    
    def to_string(self,param:dict):
        
        segments=[]
        for key ,val in param.items():
            assert self.con not in key
            assert self.sepa not in key
            seg="{}{}{}".format(key,self.con,val)
            segments.append(seg)
        return ",".join(segments)
    
    
    def get_value(self,key):
        
        segments=self.stri.split(self.sepa)
        for seg in segments:
            if seg.startswith(key):
                return seg.split(self.con)[1]
                loc=len(key)+len(self.con)
                return seg[loc:]
        return None

    def to_dict(self):
        
        segments=self.stri.split(self.sepa)
        
        mapping={}
        for seg in segments:
            key,val=seg.split(self.con)
            mapping[key]=val
        return mapping