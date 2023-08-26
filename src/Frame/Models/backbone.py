

import torch
import torch.nn as nn


from torchvision.models import resnet18,resnet34,resnet50,resnet101,resnet152

import efficientnet_pytorch


class BackBoneBase(nn.Module):
    
    
    def __init__(self,model:nn.Module,num_feat:int):
        super(BackBoneBase,self).__init__()
        
        self.model=model
        self.num_feat=num_feat
        
        
    def forward_func(self,x):
        raise NotImplementedError

    
    def forward(self,x):
   
        features=self.forward_func(x)
        assert(len(features)==self.num_feat)
        return features
        

class ResnetBackbone(BackBoneBase):
    
    
    def __init__(self,resnet_name,pretrained=True,num_feat=1):
        assert(resnet_name in ["resnet18","resnet34","resnet50","resnet101","resnet152"])
        model=eval(resnet_name)(pretrained=pretrained)
        super(ResnetBackbone,self).__init__(model,num_feat)
         
    def forward_func(self,x):
        x = self.model.conv1(x)
        x =  self.model.bn1(x)
        x =  self.model.relu(x)
        x =  self.model.maxpool(x)
        x1 =  self.model.layer1(x)
        x2 =  self.model.layer2(x1)
        x3 =  self.model.layer3(x2)
        x4 =  self.model.layer4(x3)
        features=[x1,x2,x3,x4]
        return features[(len(features)-self.num_feat):]
    
    
    
#https://github.com/lukemelas/EfficientNet-PyTorch
    
    
class EfficientBackBone(BackBoneBase):
    
    def __init__(self,efficient_name,pretrained=True,num_feat=1):
          
        assert(efficient_name in ["efficientnet-b0","efficientnet-b0","efficientnet-b1","efficientnet-b2", "efficientnet-b3","efficientnet-b4"])  
        assert(pretrained==True)
        assert(num_feat==1)
        model=EfficientNet.from_pretrained(efficient_name)
        super(EfficientBackBone,self).__init__(model,num_feat)
    
    
    def forward_func(self, x):
        # Convolution layers
        x = [self.model.extract_features(x)]
        return x
    