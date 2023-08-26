


from .backbone import *
from .unet import Unet_Decoder






class AnomalyNet(nn.Module): 
    
    
    
    def __init__(self,num_classes,num_feat=3,base_channels=64,**kwargs):
        super(AnomalyNet,self).__init__()
        
        self.backbone=ResnetBackbone("resnet18",pretrained=True,num_feat=num_feat) 
        
      
        self.unet_head=Unet_Decoder(n_classes=num_classes,base_channels=base_channels,level=num_feat)
        
        self.neck=nn.Sequential(*[nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten()])
        self.train_mode=kwargs.get("train_mode",True)
        
    
    def forward(self,x):
        if self.train_mode:
            features=self.backbone(x)
            # for f in features:
            #     print(f.shape)
            out=self.unet_head(features).permute(0,2,3,1)
            return out
        else:
            features=self.backbone(x)
            embed=self.neck(features[-1])
            out=self.unet_head(features)
            return embed, out
    
    
    def freeze_backbone(self,freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = (not freeze)
        

    def set_train_mode(self,mode):
        self.train_mode=mode
        
        
        
class MultiHeadAnmolayNet(AnomalyNet): 
    
    def __init__(self,
    