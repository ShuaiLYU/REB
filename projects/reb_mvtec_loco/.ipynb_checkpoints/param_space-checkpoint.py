import src.Frame.exp_tool as ET
from src.global_param import  MvtecLOCO

base=ET.Param()
base.device="cuda"
base.gpu_id="0"
base.mvt_data=ET.Param(root_dir=MvtecLOCO.data_dir,all_types=MvtecLOCO.all_types,size=256,mode="train")
base.all_types = MvtecLOCO.all_types


# base.mvt_data.suffix=".jpg"  # mtd dataset use  .jpg suffix for image and  png for mask img
base.mvt_data_eval=ET.Param(**base.mvt_data)
base.mvt_data_eval.mode="test"

base.num_classes=7
base.dm=ET.Param(func_name="")
base.collate_func=ET.Param()
base.collate_func.key_label_map={"img":0,"img_cp":1,"img_cps":2}
base.collate_func.data_format=["img","class"]


base.train_loader=ET.Param()
base.train_loader.batch_size=96
base.train_loader.drop_last=True
base.train_loader.shuffle=True
base.train_loader.num_workers=8
base.train_loader.persistent_workers=True
base.train_loader.pin_memory=True
base.train_loader.prefetch_factor=5


base.optim=ET.get_common_SGD_optim_param("SGD")
base.optim.SGD.lr=0.03
base.optim.SGD.weight_decay=0.00003
base.optim.lrReducer_name="CosineAnnealingWarmRestarts"
base.optim.CosineAnnealingWarmRestarts=ET.Param( T_0=510, T_mult=1, eta_min=1e-5, last_epoch=-1, verbose=False)

base.run=ET.Param(epoches=500,num_steps_per_epoch=1,epo_grad_accumulate=False)




base.hook=ET.Param()
base.hook.names=["AdjustEpochResHook","CheckPointHook"]
base.hook.AdjustEpochResHook=ET.Param(freeze_epoch=40)


base.hook.CheckPointHook=ET.Param(save_period=100,
                            model_name="resnet18",
                            extra_keys=[])


#####################################################################################




# exper=base.clone()
# exper.collate_func=ET.Param()
# exper.collate_func.key_label_map={"img":0,"img_bezier_cp":1,"img_bezier_scar_cp":2,"img_bezier_clump_cp":3
#                                  ,"img_bezier_noise":4,"img_bezier_scar_noise":5,"img_bezier_clump_noise":6}
# exper.dm=ET.Param(func_name="dm_com6")
# exper.hook.CheckPointHook=ET.Param(save_period=300,model_name="wide_resnet50_2",extra_keys=[])
# exper.num_classes=7
# set_train_param(exper,4)
# dm_com6_wr50=exper
#
#
#
# exper=base.clone()
# exper.collate_func=ET.Param()
# exper.collate_func.key_label_map={"img":0,"img_bezier_cp":1,"img_bezier_scar_cp":2,"img_bezier_clump_cp":3
#                                  ,"img_bezier_noise":4,"img_bezier_scar_noise":5,"img_bezier_clump_noise":6}
# exper.dm=ET.Param(func_name="dm_com6")
# exper.hook.CheckPointHook=ET.Param(save_period=300,model_name="resnet18",extra_keys=[])
# exper.num_classes=7
# set_train_param(exper,1)
# dm_com6_res18=exper
#
# dm_com6_res18_bs600=dm_com6_res18.clone()
# set_train_param(dm_com6_res18_bs600,3,600,300)
#
# exper=dm_com6_res18.clone()
# set_train_param(exper,1,196,500)
# exper.optim.CosineAnnealingWarmRestarts.T_0=int(500*2)
# exper.collate_func.key_label_map={"img":0,"img_bezier_cp":1,"img_bezier_scar_cp":2,"img_bezier_clump_cp":3
#                                  ,"img_bezier_noise":4,"img_bezier_scar_noise":5,"img_bezier_clump_noise":6}
# exper.num_classes=7
# exper.hook.CheckPointHook=ET.Param(save_period=500,model_name="resnet18",extra_keys=[])
# exper.network="resnet18"
# dm_com6_res18_bs600_epo2000=exper
#
#
#
# ##########
# exper=base.clone()
# exper.collate_func=ET.Param()
# exper.collate_func.key_label_map={"img":0,"img_bezier_cp":1,"img_bezier_scar_cp":2,"img_bezier_clump_cp":3
#                                  ,"img_bezier_noise":4,"img_bezier_scar_noise":5,"img_bezier_clump_noise":6}
# exper.dm=ET.Param(func_name="dm_com6")
# exper.hook.CheckPointHook=ET.Param(save_period=300,model_name="resnet101",extra_keys=[])
# exper.num_classes=7
# set_train_param(exper,4,512,300)
# dm_com6_res101=exper
#


def set_train_param(exper,num_iter_bofer_update_grad=1,batch_size=480,epoches=300):
    exper.train_loader.batch_size = int(batch_size/ num_iter_bofer_update_grad/ len(exper.collate_func.key_label_map))
    exper.run = ET.Param(epoches=epoches, num_steps_per_epoch=num_iter_bofer_update_grad,
                         epo_grad_accumulate=(num_iter_bofer_update_grad>1))
    exper.optim.CosineAnnealingWarmRestarts.T_0=int(300*1.1)
    exper.hook.CheckPointHook=ET.Param(save_period=300,model_name=exper.network,extra_keys=[])
    exper.weight="last"



def set_dm6_param(param,num_forward_bofore_backward=4):
    param=param.clone()
    param.collate_func=ET.Param()
    param.collate_func.key_label_map={"img":0,"img_bezier_cp":1,"img_bezier_scar_cp":2,"img_bezier_clump_cp":3
                                     ,"img_bezier_noise":4,"img_bezier_scar_noise":5,"img_bezier_clump_noise":6}
    param.num_classes=7
    param.dm=ET.Param(func_name="dm_com6")

    # set_train_param(exper,3,640,300)
    set_train_param(param,num_forward_bofore_backward,256*4,300)

    return param



vgg11_imagenet=base.clone()
vgg11_imagenet.network="vgg11_bn"
vgg11_imagenet.weight="ImageNet"

vgg11_dm_com6_bs1024_epo300=set_dm6_param(vgg11_imagenet,3)



exper=base.clone()
exper.network="resnet18"
exper.weight="ImageNet"
res18_imagenet=exper


exper=base.clone()
exper.network="wide_resnet50_2"
exper.weight="ImageNet"
wr50_imagenet=exper

exper=base.clone()
exper.network="wide_resnet101_2" #,"wide_resnet101_2"]:
exper.weight="ImageNet"
wr101_imagnet=exper



res18_imagenet_fs01=res18_imagenet.clone()
res18_imagenet_fs01.fewshot=1

res18_imagenet_fs02=res18_imagenet.clone()
res18_imagenet_fs02.fewshot=2

res18_imagenet_fs05=res18_imagenet.clone()
res18_imagenet_fs05.fewshot=5

res18_imagenet_fs10=res18_imagenet.clone()
res18_imagenet_fs10.fewshot=10

res18_imagenet_fs20=res18_imagenet.clone()
res18_imagenet_fs20.fewshot=20

res18_imagenet_fs50=res18_imagenet.clone()
res18_imagenet_fs50.fewshot=50







res18_dm_com6_bs1024_epo300=set_dm6_param(res18_imagenet,3)


res18_dm_com6_bs512_epo300=res18_dm_com6_bs1024_epo300.clone()
set_train_param(res18_dm_com6_bs512_epo300, 2, 512, 300)

res18_dm_com6_bs512_epo300_fs01=res18_dm_com6_bs512_epo300.clone()
res18_dm_com6_bs512_epo300_fs01.fewshot=1

res18_dm_com6_bs512_epo300_fs02=res18_dm_com6_bs512_epo300.clone()
res18_dm_com6_bs512_epo300_fs02.fewshot=2

res18_dm_com6_bs512_epo300_fs05=res18_dm_com6_bs512_epo300.clone()
res18_dm_com6_bs512_epo300_fs05.fewshot=5

res18_dm_com6_bs512_epo300_fs10=res18_dm_com6_bs512_epo300.clone()
res18_dm_com6_bs512_epo300_fs10.fewshot=10

res18_dm_com6_bs512_epo300_fs20=res18_dm_com6_bs512_epo300.clone()
res18_dm_com6_bs512_epo300_fs20.fewshot=20

res18_dm_com6_bs512_epo300_fs50=res18_dm_com6_bs512_epo300.clone()
res18_dm_com6_bs512_epo300_fs50.fewshot=50


wr50_dm_com6_bs1024_epo300=set_dm6_param(wr50_imagenet,7)


def set_draem_param(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map = {"img": 0, "img_draem": 1}
    param.num_classes = 2
    param.dm = ET.Param(func_name="get_draem_wrapper")
    set_train_param(param,num_forward_bofore_backward,256*4,300)

    return param
def set_cut_paste(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map = {"img": 0, "img_cp": 1,"img_cps": 2}
    param.num_classes = 3
    param.dm = ET.Param(func_name="dm_cm2_cp3way")
    set_train_param(param,num_forward_bofore_backward,256*4,300)
    return param

res18_draem_bs1024_epo300=set_draem_param(res18_imagenet,2)

res18_cutpaste_bs1024_epo300=set_cut_paste(res18_imagenet,2)

def set_dm3_noise(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map = {"img": 0, "img_bezier_noise": 1, "img_bezier_scar_noise": 2,
                                        "img_bezier_clump_noise": 3}
    param.num_classes = len(param.collate_func.key_label_map)
    param.dm = ET.Param(func_name="dm_com3_noise")
    set_train_param(param,num_forward_bofore_backward,256*4,300)
    return param

res18_dm3_noise_bs1024_epo300=set_dm3_noise(res18_imagenet,3)

def set_dm3_cp(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map = {"img":0,"img_bezier_cp":1,"img_bezier_scar_cp":2,"img_bezier_clump_cp":3}
    param.num_classes = len(param.collate_func.key_label_map)
    param.dm = ET.Param(func_name="dm_com3_cp")
    set_train_param(param,num_forward_bofore_backward,256*4,300)
    return param

res18_dm3_cp_bs1024_epo300=set_dm3_cp(res18_imagenet,3)


def set_dm2_bezier(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map = {"img":0,"img_bezier_cp":1,"img_bezier_noise":2}
    param.num_classes = len(param.collate_func.key_label_map)
    param.dm = ET.Param(func_name="dm_com2_bezier")
    set_train_param(param,num_forward_bofore_backward,256*4,300)
    return param

res18_dm2_bezier_bs1024_epo300=set_dm2_bezier(res18_imagenet,3)


def set_dm_com2_bezierscar(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map ={"img":0,"img_bezier_scar_noise":1,"img_bezier_scar_cp":2}
    param.num_classes = len(param.collate_func.key_label_map)
    param.dm = ET.Param(func_name="dm_com2_bezierscar")
    set_train_param(param,num_forward_bofore_backward,256*4,300)
    return param

res18_dm_com2_bezierscar_bs1024_epo300=set_dm_com2_bezierscar(res18_imagenet,3)


def set_dm_com2_bezierclump(param,num_forward_bofore_backward):
    param = param.clone()
    param.collate_func = ET.Param()
    param.collate_func.key_label_map ={"img":0,"img_bezier_clump_cp":1,"img_bezier_clump_noise":2}
    param.num_classes = len(param.collate_func.key_label_map)
    param.dm = ET.Param(func_name="dm_com2_bezierclump")
    set_train_param(param,num_forward_bofore_backward,256*4,300)
    return param

res18_dm_com2_bezierclump_bs1024_epo300=set_dm_com2_bezierclump(res18_imagenet,3)

















