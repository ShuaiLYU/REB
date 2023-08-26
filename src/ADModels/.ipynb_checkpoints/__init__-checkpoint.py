

from .ad_network import *
from .ad_head import  *
from .ad_neck import  *
from .feature_extractor import  *

import src.Frame.exp_tool as ET

def get_extract_layer(args):
    def _get_extract_layer(model_name, flag):
        if model_name == "vgg11_bn":
            if flag == "L23":
                return ["features.15", "features.22"]
            if flag == "L2":
                return ["features.15"]
            if flag == "L3":
                return ["features.22"]

        if model_name in ["resnet18","wide_resnet50_2","resnet101","wide_resnet101_2"]:
            if flag == "L23":
                return ["layer2", "layer3"]
            if flag == "L2":
                return ["layer2"]
            if flag == "L34":
                return ["layer3", "layer4"]
            if flag == "L3":
                return ["layer3"]
            if flag == "L4":
                return ["layer4"]
        return None

    layer = _get_extract_layer(args.network, args.layer)
    assert (layer is not None)
    print("layer names...")
    print(layer)
    return layer


def get_embed_len(args):
    if args.network in ["vgg11_bn", "resnet18"]:
        return 512
    if args.network in ["wide_resnet50_2"]:
        return 1024
    if args.network in ["resnet101","wide_resnet101_2"]:
        return 1024


def get_feature_len(args):
    if args.network in ["vgg11_bn", "resnet18"]:
        return 512
    if args.network in ["wide_resnet50_2"]:
        return 2048
    if args.network in ["resnet101","wide_resnet101_2"]:
        return 2048



def get_neck(args):
    num_layers = len(get_extract_layer(args))
    if args.ad_method == "gde":
        return "AvgPoolAdNeck", dict(target_embed_dimension=get_embed_len(args))
    # else:
    #     neck_param = dict(in_channels=get_feature_len(args),
    #                       out_channels=get_embed_len(args))
    #     return "AdapterAdNeck",neck_param

    else:
        embed_len = get_embed_len(args)
        neck_param = dict(pretrain_embed_dimension=embed_len,
                          target_embed_dimension=embed_len, ref_size_index=0)
        return "MulitLayerAggreAdNeck", neck_param


def get_head(args):
    def _get_head(args):
        assert (args.coreset > 0 and args.coreset <= 1)
        if args.ad_method == "gde":
            return "GdeAdHead", dict()

        else:
            head_name = "NNHead"
            nn_types={"knn":0,"kthnn":1,"lof":2,"ldof":3,"ldknn":4}
            head_param = dict(n_nearest_neighbours=1,coreset_percent=args.coreset,nn_type=NN_Type.knn)
            return head_name,head_param
        #
        # if args.ad_method == "knn":
        #     if args.coreset == 1:
        #         head_name = "KnnAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k)
        #     else:
        #         head_name = "CoresetKnnAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k,
        #                           coreset_percent=args.coreset)
        #     return head_name, head_param
        # if args.ad_method == "kthnn":
        #     if args.coreset == 1:
        #         head_name = "KthnnAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k)
        #     else:
        #         head_name = "CoresetKthnnAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k,
        #                           coreset_percent=args.coreset)
        #     return head_name, head_param
        # if args.ad_method == "ldknn":
        #     if args.coreset == 1:
        #         head_name = "LdnKnnAdHead"
        #         head_param = dict(n_nearest_neighbours=1,
        #                           n_reach_field=args.knn_k, ldn_factor=args.ldknn_factor)
        #     else:
        #         head_name = "CoreSetLdnKnnAdHead"
        #         head_param = dict(n_nearest_neighbours=1, n_reach_field=args.knn_k,
        #                           ldn_factor=args.ldknn_factor,
        #                           coreset_percent=args.coreset)
        #     return head_name, head_param
        # if args.ad_method == "lof":
        #     if args.coreset == 1:
        #         head_name = "LofAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k)
        #     else:
        #         head_name = "CoresetLofAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k,
        #                           coreset_percent=args.coreset)
        #     return head_name, head_param
        # if args.ad_method == "ldof":
        #     if args.coreset == 1:
        #         head_name = "LdofAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k)
        #     else:
        #         head_name = "CoresetLdofAdHead"
        #         head_param = dict(n_nearest_neighbours=args.knn_k,
        #                           coreset_percent=args.coreset)
        #     return head_name, head_param

    head_name, head_param = _get_head(args)
    # print(  head_name,head_param)
    return head_name, head_param


def get_ad_method_alias(args):
    ad_method_name = ""
    ad_method_name = ad_method_name + "_network" + str(args.network)
    ad_method_name = ad_method_name + "_weight" + str(args.weight)
    ad_method_name = ad_method_name + "_layer" + str(args.layer)
    ad_method_name = ad_method_name + "_" + str(args.ad_method)
    # ad_method_name = ad_method_name + "_ad" + str(args.ad_method)
    # ad_method_name = ad_method_name + "_coreset" + str(args.coreset)
    # ad_method_name = ad_method_name + "_k" + str(args.k)

    return ad_method_name

def get_ad_method_param(param):
    assert hasattr(param,"network")
    assert hasattr(param, "layer")
    assert hasattr(param, "imagesize")
    param_tree=ET.Param(
        input_shape=[3,param.imagesize[1],param.imagesize[0]],
        method_alias=get_ad_method_alias(param),
        extract_layer_name=get_extract_layer(param),
        neck_name=get_neck(param)[0], neck_param=get_neck(param)[1],
        head_name=get_head(param)[0], head_param=get_head(param)[1])
    return param_tree

# def get_ad_method_param(param):
#     assert hasattr(param,"network")
#     assert hasattr(param, "layer")
#     assert hasattr(param, "weights")
#     assert hasattr(param, "resize")
#     assert hasattr(param, "imagesize")
#     assert hasattr(param, "ad_method")
#     if param.ad_method=="ldknn":
#         assert hasattr(param, "ldknn_factor")
#     assert hasattr(param, "knn_k")
#     assert hasattr(param, "coreset")
#     assert(isinstance(param.imagesize,int))
#     param_tree=ET.Param(
#         weight_name=param.weights,
#
#         input_shape=[3,param.imagesize,param.imagesize],
#         method_alias=get_ad_method_alias(param),
#         extract_layer_name=["neck"],
#         neck_name="IdentityAdNeck", neck_param=dict(),
#         # extract_layer_name=["layer2","layer3"],
#         # neck_name=get_neck(param)[0], neck_param=get_neck(param)[1],
#
#         head_name=get_head(param)[0], head_param=get_head(param)[1])
#     return param_tree




def get_ad_method_wrapper(param_tree):
    def ad_method_wrapper(backbone, device):
        # print(backbone, param_tree.extract_layer_name)
        exactor = FeatureExtractor(backbone, param_tree.extract_layer_name)
        param_tree.neck_param["feature_dimensions"] = exactor.feature_dimensions(param_tree.input_shape, device)
        neck = eval(param_tree.neck_name)(**param_tree.neck_param)
        head = eval(param_tree.head_name)(device=device, **param_tree.head_param)
        ad_net = AdNetwork()
        ad_net.load(exactor, neck, head)
        ad_net.to(device)
        return ad_net
    return ad_method_wrapper

def get_proxy_net_param(param):

    assert hasattr(param,"network")
    assert hasattr(param, "pretrained")
    assert hasattr(param, "imagesize")
    proxy_net = ET.Param()
    proxy_net.backbone_name=param.network
    proxy_net.pretrained = param.pretrained
    proxy_net.input_shape = [3, param.imagesize[1], param.imagesize[0]]
    # proxy_net.layers_to_extract_from = get_extract_layer(param)

    proxy_net.layers_to_extract_from=["layer4"]
    proxy_net.neck_name,   proxy_net.neck_param ="AvgPoolAdNeck", dict(target_embed_dimension=get_feature_len(param))
    proxy_net.head_name = "DenseHead"


    # proxy_net.layers_to_extract_from = ["layer2","layer3"]
    # proxy_net.neck_name,   proxy_net.neck_param = get_neck(param)
    # proxy_net.head_name="AvgDenseHead"


    return proxy_net

def get_proxy_net_wrapper(param_tree):
    def net_wrapper(num_classes, device):
        proxy_net=AdNetworkSSL(param_tree.backbone_name,
                               param_tree.layers_to_extract_from,device,param_tree.input_shape,
                               param_tree.pretrained)
        proxy_net.build_neck(param_tree.neck_name,param_tree.neck_param)

        head_param = ET.Param(head_layers=[512,256], num_classes=num_classes)
        proxy_net.build_head(param_tree.head_name, head_param)
        proxy_net.to(device)
        return proxy_net
    return net_wrapper