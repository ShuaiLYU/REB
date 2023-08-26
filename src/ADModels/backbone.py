import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,resnet50,vgg11,vgg11_bn,resnet101


"""
cul the enom  with the train_embeds and    group normalize the train_embeds and test_embeds

"""

def group_normalize(train_embeds,test_embeds, p=2, dim=1, eps=1e-12):
        enom = train_embeds.norm(p, dim, True).clamp_min(eps)
        return train_embeds/enom.expand_as(train_embeds),test_embeds/enom.expand_as(test_embeds)
    

    
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
   
    




    
def resnet_forword(model,inputs):
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x

def vgg_forward(model, x):
    x = model.features(x)

    return x



import efficientnet_pytorch
class EfficientNet(efficientnet_pytorch.EfficientNet):
    
    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs)
        return x
    



def get_backbone(backbone_name,**kwargs):
    assert(backbone_name in ["resnet18","resnet50",'wide_resnet50_2','wide_resnet101_2'
                             "resnet101",
                             "vgg11","vgg11_bn",
                             "efficientnet-b0",
                             "efficientnet-b0","efficientnet-b1",
                             "efficientnet-b2", "efficientnet-b3",
                             "efficientnet-b4"],backbone_name)
    
    pretrained=kwargs.get("pretrained",True)
    if "resnet" in  backbone_name:

        if pretrained:
            print (  "init resnet from imagenet")
        else:
            print (  "init resnet from scratch")
        if "wide_resnet" in backbone_name:
            print("init {} !!!".format(backbone_name))
            model = torch.hub.load('pytorch/vision:v0.13.1', backbone_name, pretrained=pretrained)
        else:
            model=eval(backbone_name)(pretrained=pretrained)
        # if backbone_name=="resnet18": embeds_len=512 
        # if backbone_name=="resnet50": embeds_len=2048
        embeds_len=model.fc.in_features
        model.fc = nn.Identity()
        model.avgpool=nn.Identity()
        return model,embeds_len
    elif "efficientnet" in backbone_name:
        model = EfficientNet.from_pretrained(backbone_name)
        embeds_len=model._fc.in_features
        return model,embeds_len
    
    elif "vgg" in backbone_name:
        model= eval(backbone_name)(pretrained=pretrained)
        model.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        model.classifier=nn.Identity()
        embeds_len=512
        return model,embeds_len