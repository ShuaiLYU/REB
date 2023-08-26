
from tqdm import tqdm
import math

    
    
    

class TrainerBase(object):
       
    def __init__(self):
        self._hooks = [] 
        self.epo=0
        self.epo_step=0
        self.step_variables={}
        self.epo_variables={}
    def register_hooks(self, hooks):
        # 将钩子注册到trainer中，实际上就是放到trainer的_hooks列表里以便后续调用
        
        
        if isinstance(hooks,HookBase):
            hooks=[hooks]
        for hook in hooks:
            # 这里为每个钩子创建一个类变量，指向当前trainer。
            # 这样就可以访问trainer内部的model、optimizer、epoch，iter等。
            assert(isinstance(hook,HookBase))
            hook.set_trainer(self)
            self._hooks.append(hook)
            
    def before_train(self):
        for hook in self._hooks:
            hook.before_train()   
    def after_train(self):   
        for hook in self._hooks:
            hook.after_train()    
    def before_epoch(self):
        for hook in self._hooks:
            hook.before_epoch()  
    def after_epoch(self):
        for hook in self._hooks:
            hook.after_epoch()  
    def before_step(self):
        for hook in self._hooks:
            hook.before_step()  
    def after_step(self):      
        for hook in self._hooks:
            hook.after_step()   
    def train_step(self):
        pass

    def run_train(self,epoches=None,num_steps_per_epoch=None):
        
        self.epoches=epoches
        self.num_steps_per_epoch=num_steps_per_epoch
        
        #0. before train
        self.before_train()

        for step in tqdm(range(self.epoches*self.num_steps_per_epoch)):
            
            self.epo = math.ceil((step+1) / self.num_steps_per_epoch)
            self.epo_step=step%self.num_steps_per_epoch+1
            self.step=step+1
            #2. train loop
            if self.epo_step==1: self.before_epoch()
            
            self.before_step()
            
            self.train_step()
            
            self.after_step()

             #the last iteration within a epoch 
            if self.epo_step==self.num_steps_per_epoch:  self.after_epoch()
        self.after_train()


class HookBase:
    
    def __init__(self):
        self.trainer=None

    def set_trainer(self, trainer: TrainerBase):
        self.trainer=trainer

    def before_train(self) -> None:
        """Called before the first iteration."""
        pass

    def after_train(self) -> None:
        """Called after the last iteration."""
        pass

    def before_epoch(self) -> None:
        """Called before each epoch."""
        pass

    def after_epoch(self) -> None:
        """Called after each epoch."""
        pass

    def before_step(self) -> None:
        """Called before each iteration."""
        pass

    def after_step(self) -> None:
        """Called before each iteration."""
        pass




class VariableBase(object):


    def __init__(self):
        self.__var_dict= {}
    def __add__(self,var_dict:dict):
        for k ,val in var_dict.items():
            if k not in self.__var_dict.keys():
                self.__var_dict[k]=list()
            self.__var_dict[k]+=val

    def get(self) -> dict:
        return  self.__var_dict
class MetricBase(object):

    def __init__(self):
        self.labels = []
        self.scores = []

    def reset(self):
        self.labels = []
        self.scores = []

    def add_batch(self, scores, labels):
        self.scores.append(scores.flatten())
        self.labels.append(labels.flatten())

    def get(self) -> dict:
        pass


class DatasetBase(object):
    
    
    def __init__(self):
        
        pass
    
    
            
class ModelBase(object):
    
    
    def __init__(self):
        
        pass