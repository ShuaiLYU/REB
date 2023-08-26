

__all__ =['AdNeckBase','IdentityAdNeck',"AvgPoolAdNeck","MaxPoolAdNeck",
          "MulitLayerAggreAdNeck","MulitLayerAggreAvgPoolAdNeck","AdapterAdNeck"]

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)



class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class AdNeckBase(torch.nn.Module):

    """

     output:  [b,h,w,c]  or  [b,1,1,c]    
    """
    def __init__(self):
        super(AdNeckBase, self).__init__()

    def get_feature_channel(self):
        pass



class IdentityAdNeck(AdNeckBase):


    def __init__(self,**kwargs):
        super(IdentityAdNeck, self).__init__()
    def  forward(self,features:list):
        assert (len(features)==1)
        return features[0]

# class IdentityAdNeck(AdNeckBase):
#
#
#     def __init__(self,layer_name):
#         super(IdentityAdNeck, self).__init__()
#         self.layer_name=layer_name
#     def  forward(self,features:list):
#         assert(len(features)==1)
#         return features[self.layer_name].permute(0,2,3,1)


    
class AvgPoolAdNeck(AdNeckBase):
    def __init__(self,target_embed_dimension=None,**kwargs):
        super(AvgPoolAdNeck, self).__init__()
        # self.layer_name=layer_name
        self.target_embed_dimension=target_embed_dimension
        self.avg=nn.AdaptiveAvgPool2d((1, 1))
    def  forward(self,features:list):
        assert(len(features)==1)
        embed= self.avg(features[0])
        return embed.permute(0,2,3,1)

    def get_feature_channel(self):
        return self.target_embed_dimension

class MaxPoolAdNeck(AdNeckBase):
    def __init__(self,**kwargs):
        super(MaxPoolAdNeck, self).__init__()
        # self.layer_name=layer_name
        self.max=nn.AdaptiveMaxPool2d((1, 1))
    def  forward(self,features:list):
        assert(len(features)==1)
        embed= self.max(features[0])
        return embed.permute(0,2,3,1)
    
    

class  MulitLayerAggreAdNeck(AdNeckBase):

    def __init__(self,feature_dimensions,pretrain_embed_dimension,target_embed_dimension,
                 patchsize=3,patchstride=1,ref_size_index=0):
        super(MulitLayerAggreAdNeck, self).__init__()
        # self.device=device
        # patchsize=3
        # patchstride=1
        # pretrain_embed_dimension=0
        # feature_dimensions= [256,512,] #每个特征的通道数
        # target_embed_dimension=512,
        # print(feature_dimensions,pretrain_embed_dimension,target_embed_dimension,patchsize,patchstride,ref_size_index)
        self.target_embed_dimension=target_embed_dimension
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dimension )
        self.preadapt_aggregator = Aggregator(target_dim=target_embed_dimension)
        self.ref_size_index=ref_size_index
        # _ = self.preadapt_aggregator.to(self.device)

    def get_feature_channel(self):
        return self.target_embed_dimension
    
    
    def  forward(self,features:list,detach=True,provide_patch_shapes=False):

        assert(len(features)>0)

        # features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        
        ref_size_index=self.ref_size_index if self.ref_size_index!=-1 else len(patch_shapes)-1
        ref_num_patches = patch_shapes[self.ref_size_index]
        
        
  
        for i in range(0, len(features)):
            if i != ref_size_index:
                _features = features[i]
                patch_dims = patch_shapes[i]

                # TODO(pgehler): Add comments

                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )       # [b,h,w,c,3,3]
                _features = _features.permute(0, -3, -2, -1, 1, 2) # [b,c,3,3,h,w]
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])  # [b*c*3*3,h,w]
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )  #[b,c,3,3,h,w]
                _features = _features.permute(0, -2, -1, 1, 2, 3)   #[b,h,w,c,3,3]
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])  # [b,h*w,c,3,3]
                features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]  # [b*h*w,c,3,3]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.preprocessing(features)           #[B*h*w,1,pretrain_embed_dimension  ]
        features = self.preadapt_aggregator(features)        # batchsize x number_of_layers x input_dim -> batchsize x target_dim


        def _detach(features):
            return features.detach().cpu().numpy() 
            # if detach:
            #     return [x.detach().cpu().numpy() for x in features]
            # return features
        features=features.reshape(-1,*ref_num_patches,*features.shape[1:])

        return features
#         if provide_patch_shapes:
#             return _detach(features).reshape(-1,ref_num_patches,features.shape[2:]), patch_shapes

#         return _detach(features)



class  MulitLayerAggreAvgPoolAdNeck(MulitLayerAggreAdNeck):

    def  forward(self,features:list,detach=True,provide_patch_shapes=False):
        features =MulitLayerAggreAdNeck.forward(self,features,detach,provide_patch_shapes)
        return features.mean(dim=[1,2],keepdim=False)


class  MulitLayerAggrCropdNeck(MulitLayerAggreAdNeck):

    def  forward(self,features:list,detach=True,provide_patch_shapes=False):
        features =MulitLayerAggreAdNeck.forward(self,features,detach,provide_patch_shapes)
        return features.mean(dim=[1,2],keepdim=False)







import torch




class ConvBlock(nn.Module):
    """conv-norm-relu"""
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, norm_layer=None):
        """
        :param in_channels:  输入通道
        :param out_channels: 输出通道
        :param kernel_size: 默认为3
        :param padding:    默认为1，
        :param norm_layer:  默认使用 batch_norm
        """
        super(ConvBlock,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else  nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.convblock(x)


class UNetBlock(nn.Module):
    """conv-norm-relu,conv-norm-relu"""
    def __init__(self, in_channels, out_channels,mid_channels=None,padding=1, norm_layer=None):
        """
        :param in_channels:
        :param out_channels:
        :param mid_channels:  默认 mid_channels==out_channels
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        """
        super(UNetBlock,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.unetblock=nn.Sequential(
            ConvBlock(in_channels,mid_channels,padding=padding,norm_layer=norm_layer),
            ConvBlock(mid_channels, out_channels,padding=padding,norm_layer=norm_layer)
        )
    def forward(self, x):
        return self.unetblock(x)



class UNetUpBlock(nn.Module):
    """Upscaling then unetblock"""

    def __init__(self, in_channels, out_channels,padding=1,norm_layer=None, bilinear=True):
        """
        :param in_channels:
        :param out_channels:
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        :param bilinear:  使用双线性插值，或转置卷积
        """

        super(UNetUpBlock,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                UNetBlock(in_channels, in_channels//2, padding=padding, norm_layer=norm_layer),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_channels, out_channels,padding=padding,norm_layer=norm_layer)


    def crop(self,tensor,target_sz):
        _, _, tensor_height, tensor_width = tensor.size()
        diff_y = (tensor_height - target_sz[0]) // 2
        diff_x = (tensor_width - target_sz[1]) // 2
        return tensor[:, :, diff_y:(diff_y + target_sz[0]), diff_x:(diff_x + target_sz[1])]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        # x2=self.crop(x2,x1.shape[2:])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AdapterAdNeck(AdNeckBase):

    def __init__(self,in_channels, out_channels,**kwargs):
        super(AdapterAdNeck, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.adapter =UNetUpBlock(in_channels,out_channels)

    def forward(self, features: list):
        assert (len(features)==2)
        # for feature in features : print(feature.shape)
        out=self.adapter(features[1],features[0])
        return out.permute(0,2,3,1)

    def get_feature_channel(self):
        return self.out_channels