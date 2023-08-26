import sys,os
project_dir=os.path.dirname(__file__)
solution_dir=os.path.dirname(os.path.dirname(project_dir))
sys.path.append(solution_dir)
#######################################
import src.ADModels as AD
import src.Frame.exp_tool as ET
import src.Frame.file_tool as FT

from src import  global_param
from src.module import *
from src.Frame.checkpoint import CheckPoint
from src.Frame import ComposeJoint,DatasetWrapper
from src.Data.anomaly_datasets import MVTecAD

from src.Data.transforms import RandomTranspose

from src.ADModels import NN_Type
#################################################################################
import copy
import torch
import numpy as np
from torchvision import transforms
###############################################################################
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-exp_name' ,type=str, required=True)
parser.add_argument('-run_name', default=None ,type=str, required=True)
# 0 DefectMaker SSL: 1:  LDKNN   2 : DefectMaker + LDKNN
parser.add_argument('-run_mode' ,type=int, required=True, choices=[0, 1, 2])
parser.add_argument('-seed' ,type=int ,default=0)
parser.add_argument('-coreset' ,type=float ,default=1)
parser.add_argument('-ldknn_factor' ,type=float ,default=1)
args = parser.parse_args()
global_param.coreset=args.coreset
global_param.ldknn_factor=args.ldknn_factor
# #########################################################################


EXPER=ET.Experiment(save_root=global_param.SAVE_ROOT,
                    project_name=ET.get_folder_name_of_file(__file__),
                    exp_name=args.exp_name,
                    run_name=args.run_name,seed=args.seed )
EXPER.set_logger()
import param_space
PARAM=eval("param_space."+args.exp_name)
EXPER.set_param(PARAM)

##########################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = PARAM.gpu_id

EXPER.set_attribute("csv", FT.CsvLogger(EXPER.get_save_dir(), "result"))

#################################################################################

#################################################################################
EXPER.info(PARAM)

if __name__ =="__main__":

    # all_types=["splicing_connectors"]



    PARAM.run_mode = args.run_mode

    for cate_idx,cate_name in enumerate(PARAM.all_types):

        EXPER.save_dir_child= os.path.join(EXPER.get_save_dir(), cate_name)
        torch.cuda.empty_cache()

        EXPER.info(cate_name)
        EXPER.set_attribute("cul_data",cate_name)
        imagesize= global_param.get_img_size(cate_name)

        buffer_param = global_param.global_param.clone()
        buffer_param.network = PARAM.network
        buffer_param.imagesize=imagesize
        buffer_param.weight = PARAM.weight

        print(buffer_param)
        # update_data_param
        PARAM.ad_method_param = AD.get_ad_method_param(buffer_param)
        PARAM.proxy_net_param = AD.get_proxy_net_param(buffer_param)
        print(PARAM.proxy_net_param)
        PARAM.cul_type=cate_name
        PARAM.mvt_data.cate_name=cate_name
        PARAM.mvt_data_eval.cate_name=cate_name
        PARAM.mvt_data.size=imagesize
        PARAM.mvt_data_eval.size=imagesize





        tensor_transform = transforms.Compose([])
        # tensor_transform.transforms.append(transforms.Resize(size=(imagesize[1],imagesize[0])))
        # tensor_transform.transforms.append(transforms.CenterCrop(224))
        tensor_transform.transforms.append(transforms.ToTensor())
        tensor_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]))

        #

        proxy_wrapper =AD.get_proxy_net_wrapper(PARAM.proxy_net_param)
        model=proxy_wrapper(PARAM.num_classes,PARAM.device)

        train_eval_data=MVTecAD(**PARAM.mvt_data)
        train_eval_data = DatasetWrapper(train_eval_data, "img", transform=tensor_transform,
                                         new_length=PARAM.fewshot)

        eval_data = DatasetWrapper(MVTecAD(**PARAM.mvt_data_eval), ("img", "label", "mask"),
                                   transform=tensor_transform)

        #lyus_stack
        # save_dir = EXPER.get_save_dir()


        ck = CheckPoint(os.path.join(EXPER.save_dir_child, "checkpoint"), buffer_param.network)
        ck.bind_model(model)

        if PARAM.run_mode in [0,2]:
            if ck.empty():

                augment_transform = ComposeJoint()
                cj_param = 0.1
                augment_transform.append(
                    [transforms.ColorJitter(brightness=cj_param, contrast=cj_param, saturation=cj_param, hue=cj_param), None])
                augment_transform.append(RandomTranspose())
                # augment_transform.append([transforms.RandomResizedCrop(size=(256,256),scale=(0.5,1)),None])
                # augment_transform.append([RandomAugment(3), None])

                train_data = MVTecAD(**PARAM.mvt_data,transform=augment_transform)
                                     # new_length=PARAM.fewshot)
                train_data=DatasetWrapper(train_data,new_length=PARAM.fewshot)
                train_data.img_size = imagesize
                train_SSL(model,PARAM,train_data,tensor_transform)
        if PARAM.run_mode in [1,2]:

            # step 2
            assert (PARAM.weight in [ "ImageNet","last"])
            if PARAM.weight !="ImageNet":
                # ck.load_the_last()

                def load_the_last(self):
                    last_one = self.csv.get_rows()[-1]
                    weight_name = last_one["weight_name"]
                    weight_path = os.path.join(self.root, weight_name)
                    func = lambda x: "backbone" in x and "exactor" not in x
                    pthfile = {k[9:]: v for k, v in torch.load(weight_path).items() if func(k)}
                    self.model.backbone.load_state_dict(pthfile)
                    print("load weights from {}".format(weight_path))
                    pass
                load_the_last(ck)

            # testtep=True
            # if testtep:
            #     model_path="/media/lyushuai/Data/OUTPUT1/dmssl/dm_com6_run1/{}/".format(defect_name)
            #
            #     ck=CheckPoint(os.path.join(model_path,"checkpoint"),"resnet18")
            #     ck.bind_model(model)
            #     ck.load_the_last()


                # ck.load_by_name("resnet18_epo5000.pth")
            ad_method=AD.get_ad_method_wrapper(PARAM.ad_method_param)
            eval_tool=AdEval(ad_method,model.backbone,PARAM.device,imagesize=PARAM.mvt_data.size)

            after_fit=False



            eval_inputs=[]

            if buffer_param.ad_method=="gde":
                eval_hparam=dict(ad_method=buffer_param.ad_method,name="_gde")
                eval_input = {"test_data": eval_data}
                eval_inputs.append((eval_hparam, eval_input))
            elif buffer_param.ad_method=="NNHead":
                for  k  in buffer_param.ks:
                    for nn_type in buffer_param.nn_types:

                        eval_input={"test_data": eval_data}
                        eval_hparam={"nn_type":nn_type,"k":k,"ad_method":buffer_param.ad_method}
                        eval_hparam["name"] = "_{}_k{}_coreset{}".format(nn_type, k,buffer_param.coreset)
                        if nn_type==NN_Type.ldknn:
                            eval_hparam["ld_coefficient"]=buffer_param.ldknn_factor
                            eval_hparam["name"] = "_{}_k{}_ld{}_coreset{}".format(nn_type, k,buffer_param.ldknn_factor,buffer_param.coreset)
                        eval_inputs.append((eval_hparam,eval_input))

            for eval_hparam,eval_input in eval_inputs:



                res_csv_name = PARAM.ad_method_param.method_alias+eval_hparam["name"]
                res_csv = FT.CsvLogger(os.path.join(EXPER.get_save_dir(), "ad_res"), res_csv_name)

                EXPER.set_attribute("res_csv", res_csv)
                def   find_result(item,csv_data):
                    if not csv_data.exists():
                        return False
                    rows=csv_data.get_rows()
                    k ,v=item
                    for row in rows:
                        if row[k]==v:
                            return True

                    return False

                if find_result(("cul_type",cate_name),EXPER.get("res_csv")):
                    continue
                if  after_fit==False:
                    eval_tool.fit(train_eval_data)
                    eval_tool.update_hyper_param(**eval_hparam)
                    after_fit=True

                eval_tool.run_eval(**eval_input)
                if cate_idx== len(PARAM.all_types) -1:
                    rows=EXPER.get("res_csv").get_rows()
                    mean_row=copy.deepcopy(rows[-1])
                    for k in mean_row.keys():
                        if k =="cul_type":
                            mean_row[k]="mean"
                        else: 
                            mean=  np.array( [ float(r[k]) for r in rows]).mean()
                            mean_row[k]=mean
                    EXPER.get("res_csv").append_one_row(mean_row)
            eval_tool.clear()

