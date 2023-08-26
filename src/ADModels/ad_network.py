





from .feature_extractor import FeatureExtractorBase,FeatureExtractor

from .ad_neck import  AdNeckBase,MulitLayerAggreAdNeck,AdapterAdNeck,AvgPoolAdNeck

from .ad_head import  AdHeadBase,DenseHead,GdeAdHead,AvgDenseHead,MaxpDenseHead
import time

from .utils import  ModuleCP

__all__ =["AdNetwork","AdNetworkSSL","ProxyNet"]


import torch.nn as nn
import torch

from tqdm import tqdm
import torch.nn.functional as F



class AdNetwork(ModuleCP):


    def __init__(self):
        super(AdNetwork, self).__init__()

        self.buff={}
    def load(self,exactor:FeatureExtractorBase,neck:AdNeckBase,head:AdHeadBase):
        self.exactor=exactor
        self.neck=neck
        self.head=head
        
        
    def run_embed(self,inputs):

        features=self.exactor(inputs,to_list=True)
        # for feat in features: print(feat.shape)   #[b,h,w,c]
        assert (len(features)>0)
        # for feat in features:  print(feat.shape)

        
        embeddings=self.neck(features)
        # print(embeddings.shape,"embeddings")
        return embeddings
    def forward(self,inputs):
        embeddings=self.run_embed(inputs)
        torch.cuda.synchronize()
        start = time.time()
        results=self.head(embeddings)
        self.buff["head_runtime"]= time.time()-start
        # for res in results: print(res.shape,"resshape")  # [b,h,w,c]
        return results


    def fit(self,embeddings):
        self.head.train()
        self.head.fit(embeddings)        
        self.head.eval()

    def fit_dataset(self,train_data,device=None):
        self.exactor.backbone.eval()

        train_embed = []
        with torch.no_grad():
            for x in tqdm(train_data):
                if device is not None: x=x.to(device)
                embed = self.run_embed(x)
                train_embed.append(embed)
        train_embed = torch.cat(train_embed)
        self.fit(train_embed)


    """
    for self supervised learning
    """
    
from .backbone import get_backbone




class AdNetworkSSL(AdNetwork):


    def __init__(self,backbone_name,layers_to_extract_from,device,input_shape,pretrained=True,**wargs):
        super(AdNetworkSSL, self).__init__()
        self.backbone_name=backbone_name
        self.backbone,_=get_backbone(self.backbone_name,pretrained=pretrained)
        self.layers_to_extract_from=layers_to_extract_from
        self.exactor= FeatureExtractor(self.backbone,layers_to_extract_from)
        self.device=device
        self.input_shape=input_shape
        
    def load(self,neck:AdNeckBase,head:AdHeadBase):
        self.neck=neck
        self.head=head
    
    def build_neck(self,neck_name,neck_param):
        # print(nec
        neck_param["feature_dimensions"]=self.exactor.feature_dimensions(self.input_shape)
        # assert(neck_name=="MulitLayerAggreAdNeck"),neck_name
        # neck_param["ref_size_index"]=-1
        self.neck=eval(neck_name)(**neck_param)
        
    def build_head(self,head_name,head_param):
        # assert(head_name=="DenseHead")
        last_layer=self.neck.get_feature_channel()
        head_param.last_layer=last_layer
        self.head=eval(head_name)(**head_param)

    def freeze_resnet(self):
        # freez full resnet18
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        for param in self.neck.parameters():
            param.requires_grad = True
        #unfreeze head:
        for param in self.head.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        # freez full resnet18
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        #unfreeze head:
        for param in self.neck.parameters():
            param.requires_grad = True
        #unfreeze head:
        for param in self.head.parameters():
            param.requires_grad = True


class ProxyNet(AdNetworkSSL):


    def __init__(self,backbone,layers_to_extract_from,device,**wargs):
        AdNetwork.__init__(self)
        self.backbone=backbone
        self.layers_to_extract_from=layers_to_extract_from
        self.exactor= FeatureExtractor(self.backbone,layers_to_extract_from)
        self.device=device
