import sys,os


src= "/src"
sys.path.append(os.path.abspath(src))

import time
###############################################################
from src import  global_param
import src.Frame.exp_tool as ET
from src.Data.anomaly_datasets import MVTecAD,DtdDataset
from src.Frame.saver import Saver
from src.Frame.optim import OptimizerKit
from src.ce_loss import SigCELoss
from src.Frame.utils import InverseNormalize,ToNumpy

from src.Frame.base import HookBase
from src.Frame.checkpoint import CheckPointHook
from src.Frame.utils import Mapper
from src.DefectMaker import  *
from src.Frame.trainer import commonTrainer
# from src.Data.augment import RandomAugment
from src.Frame.metrics import AucMetric,BinaryClassMetric

from src.ADModels.PatchCore.common import RescaleSegmentor
from src.Frame.Image.dye import VisualTool

#####################################################################
import torch
import numpy as np
import copy
import  torch.nn as nn
from torch.utils.data import DataLoader
from  tqdm import tqdm
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
########################################











class CheckPointHookAlpha(CheckPointHook):

    def after_train(self):
        self.load_last()


class CommonMetricHook(HookBase):

    def __init__(self, dataset, metric, metric_name, period, batch_size):

        self.dataset = dataset
        self.metric_name = metric_name
        self.metric = metric
        self.period = period
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=0, drop_last=False)

    def after_epoch(self):

        epoch = self.trainer.epo
        if epoch % self.period != 0:
            return
        model = self.trainer.model
        model.eval()
        model.set_train_mode(False)
        metrics = self.eval_run(model)
        model.set_train_mode(True)
        model.train()
        # for k ,val in metrics.items():
        #     self.trainer.epo_variables[self.metric_name+k]=val
        for key, val in metrics.items():
            key = "_".join(["epo", self.metric_name, key])
            self.trainer.saver.add_scalar(key, val, self.trainer.step)

    def eval_run(self, model):
        device = self.trainer.device
        self.metric.reset()
        with torch.no_grad():
            for x_batch, y_batch in self.data_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                # print(x_batch,y_batch)
                embed_batch, logits_batch = model(x_batch)
                score_batch = 1 - torch.sigmoid(logits_batch)[:, 0]
                self.metric.add_batch(score_batch.cpu().numpy(), y_batch.cpu().numpy().astype(np.int64))

        return self.metric.get()

    def after_step(self):
        step_variables = self.trainer.step_variables
        with torch.no_grad():
            y_batch = step_variables["y_batch"]
            y_batch = torch.where(y_batch > 0, 1, 0)
            fx_batch = step_variables["fx_batch"]
            score_batch = 1 - torch.sigmoid(fx_batch)[:, 0]

            self.metric.reset()
            self.metric.add_batch(score_batch.numpy(), y_batch.numpy().astype(np.int64))
            metrics = self.metric.get()
            for key, val in metrics.items():
                key = "_".join(["train", "batch", self.metric_name, key])
                self.trainer.saver.add_scalar(key, val, self.trainer.step)


class AdjustEpochResHook(HookBase):

    def __init__(self, freeze_epoch=80):

        self.freeze_epoch = freeze_epoch

    #     def before_train(self):

    #         self.trainer.model.freeze_resnet()
    #         self.has_freezed=True

    def before_epoch(self):

        if self.trainer.epo < self.freeze_epoch:
            self.trainer.model.freeze_resnet()
        else:
            self.trainer.model.unfreeze()


#     def after_epoch(self):
#         epoch = self.trainer.epo
#         if epoch>self.freeze_epoch and self.has_freezed:
#             self.trainer.model.unfreeze()
#             self.has_freezed=False



class TrainAlpha(commonTrainer):
    def train_step(self):

        x_batch, y_batch = next(self.train_data_loader)

        # run_map can handle tensor, list(tensor) ,dict(tensor)
        x_batch = Mapper(lambda x: x.to(self.device))(x_batch)
        y_batch = Mapper(lambda x: x.to(self.device))(y_batch)
        fx_batch = self.model(x_batch)
        # update_mask_batch=mask_tensor.detach().cpu().numpy()
        ###### cul loss
        loss_dict = {}
        if isinstance(self.loss_dict, nn.Module):
            loss_dict = self.loss_dict(fx_batch, y_batch)
            if not isinstance(loss_dict, dict): loss_dict = {"loss": loss_dict}
        elif isinstance(self.loss_dict, dict):
            loss_dict = {name: loss_func(fx_batch, y_batch) for name, loss_func in self.loss_dict.items()}
        elif isinstance(self.loss_dict, list):
            loss_dict = {idx: loss_func(fx_batch, y_batch) for idx, loss_func in enumerate(self.loss_dict)}

        if not self.epo_grad_accumulate:
            batch_loss_sum = sum(loss_dict.values())
            batch_loss_sum.backward()

            self.optimizer.step(batch_loss_sum.detach().cpu().numpy(),
                                epoch_end=(self.epo_step == self.num_steps_per_epoch))
            self.model.zero_grad()
        else:
            batch_loss_sum = sum(loss_dict.values()) / self.num_steps_per_epoch
            batch_loss_sum.backward()
            backward = False
            if self.epo_step == self.num_steps_per_epoch:
                self.optimizer.step(batch_loss_sum.detach().cpu().numpy(),
                                    epoch_end=(self.epo_step == self.num_steps_per_epoch))
                self.model.zero_grad()
                backward = True
            # print(" epo {} step {} : backward : {}".format(self.epo,self.epo_step,backward))
        step_variables = {}
        # step_variables["x_batch"]=x_batch
        to_cpu = Mapper(lambda x: x.detach().cpu())
        step_variables["x_batch"] = to_cpu(x_batch)
        step_variables["y_batch"] = to_cpu(y_batch)
        step_variables["fx_batch"] = to_cpu(fx_batch)
        step_variables["loss_dict"] = to_cpu(loss_dict)
        step_variables["batch_loss_sum"] = batch_loss_sum.item()
        step_variables["epo"] = self.epo
        step_variables["lr"] = self.optimizer.get_lr()
        self.step_variables = step_variables

    def run_train(self, epoches=None, num_steps_per_epoch=None, **wargs):

        self.epo_grad_accumulate = wargs.get("epo_grad_accumulate", False)
        self.epoches = epoches
        self.num_steps_per_epoch = num_steps_per_epoch

        # 0. before train
        self.before_train()
        self.model.train()
        for step in tqdm(range(self.epoches * self.num_steps_per_epoch)):

            self.epo = math.ceil((step + 1) / self.num_steps_per_epoch)
            self.epo_step = step % self.num_steps_per_epoch + 1
            self.step = step + 1
            # 2. train loop
            if self.epo_step == 1: self.before_epoch()

            self.before_step()

            self.train_step()

            self.after_step()

            # the last iteration within a epoch
            if self.epo_step == self.num_steps_per_epoch:  self.after_epoch()
        self.after_train()


class VisualHook(HookBase):

    def __init__(self, num_epoch, mean, std):
        self.num_epoch = num_epoch
        self.inverse = InverseNormalize(mean, std)
        self.toNp = ToNumpy(transpose=True)

    def after_epoch(self):
        epoch = self.trainer.epo
        # if epoch==1 or epoch%self.num_epoch==0:
        if epoch < self.num_epoch:
            img_batch = self.trainer.step_variables["x_batch"]
            img_batch = self.inverse(img_batch)
            img_batch = self.toNp(img_batch)
            file_name_batch = [str(i) + ".jpg" for i in range(len(img_batch))]
            self.trainer.saver.visualize([img_batch], "epo_{}".format(epoch), file_name_batch)



def train_SSL(model,PARAM,train_data,tensor_transform):

    if PARAM.dm.func_name=="get_draem_wrapper":
        dm_wrapper = eval(PARAM.dm.func_name)(DtdDataset(global_param.dtd_data_root))
    else:
        dm_wrapper = eval(PARAM.dm.func_name)(train_data)

    cj_param=0.1
    # tensor_transform.transforms= [transforms.ColorJitter(brightness=cj_param, contrast=cj_param, saturation=cj_param, hue=cj_param),RandomTranspose()
    #                               ]+tensor_transform.transforms

    wrappped_data = dm_wrapper(train_data, transform=tensor_transform, new_length=2000)
    collate_func = wrappped_data.get_collate_func(**PARAM.collate_func)
    dataloader = DataLoader(dataset=wrappped_data, collate_fn=collate_func, **PARAM.train_loader)

    #
    # dataset_list=[]
    # for i in range(0,2):
    #     transform= None if i==0 else augment_transform
    #     dataset_list.append(MVTecLoco(**PARAM.mvt_data,transform=transform))
    # train_eval_data = DataListWrapper(dataset_list, "img", transform=tensor_transform)

    model_param=model.parameters()
    optimkit = OptimizerKit(parameters=model_param, **PARAM.optim)
    loss_fn = SigCELoss()
    hooks = []
    for hook_name in PARAM.hook.names:
        hook = eval(hook_name)(**PARAM.hook[hook_name])
        hooks.append(hook)
    # DensityEvalHook
    # densityAucMetricHook=DensityEvalHook(train_eval_data,eval_data,AucMetric(),"ad",period=PARAM.valid_period,batch_size=64)
    # adAucMetricHook=AdEvalHook(train_eval_data,eval_data,AucMetric(),"density",period=PARAM.valid_period,batch_size=64)
    # aucMetricHook=CommonMetricHook(eval_data,AucMetric(),"class",period=PARAM.valid_period,batch_size=64)

    visHook = VisualHook(10, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    hooks.append(visHook)
    # hooks+=[aucMetricHook]
    EXPER = ET.get_existing_exp()  # 获取单实例 试验 instance
    saver = Saver(EXPER.save_dir_child)
    trainer = TrainAlpha()
    trainer.register_hooks(hooks)
    trainer.load(model, optimkit, dataloader, saver, {"loos": loss_fn}, device=PARAM.device)
    trainer.run_train(**PARAM.run)








def get_tensor_indice(tensor, values):

    mask=torch.zeros_like(tensor)
    for val in values:
        mask =mask | torch.eq(tensor, val)
    return torch.nonzero(mask)

def get_array_indice(array,values):
    array=array.astype(np.int)
    mask = np.zeros_like(array)
    for val in values:
        mask = mask | np.equal(array, val)
    return np.nonzero(mask)


class AucMetricChild(AucMetric):
    def get(self):
        res = {}
        labels,scores=np.concatenate(self.labels), np.concatenate(self.scores)

        fpr, tpr, _ = roc_curve(np.where(labels> 0, 1, 0),scores)
        roc_auc = auc(fpr, tpr)
        res["roc_auc"] = roc_auc
        if np.unique(labels).size>2:
            for i in range(1,np.unique(labels).size):
                indices=get_array_indice(labels,[0,i])
                labels_i,scores_i=labels[indices], scores[indices]
                fpr, tpr, _ = roc_curve(np.where(labels_i > 0, 1, 0),scores_i)
                res["roc_auc_class{}".format(i)] =  auc(fpr, tpr)

        sel_idxs = []
        for i in range(len(self.labels)):
            if np.sum(self.labels[i]) > 0:
                sel_idxs.append(i)
        labels, scores=np.concatenate([self.labels[i] for i in sel_idxs]),np.concatenate([self.scores[i] for i in sel_idxs])
        fpr, tpr, _ = roc_curve(np.where(labels > 0, 1, 0), scores)
        res["anomaly_roc_auc"] = roc_auc = auc(fpr, tpr)

        return res

class AdEval(object):
    def __init__(self,ad_method,backbone,device,imagesize):

        self.metric=AucMetricChild()
        self.class_metric=BinaryClassMetric()

        self.imagesize=imagesize

        self.visual_Tool=VisualTool( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.device = device

        self.anomaly_segmentor = RescaleSegmentor(
            device=device, target_size=(self.imagesize[1],self.imagesize[0])
        )

        torch.cuda.empty_cache()
        # backbone_cp = copy.deepcopy(backbone)
        backbone.to(device)
        backbone.eval()
        ###############################
        self.ad_method=ad_method
        self.ad_net = self.ad_method(backbone, device)
        self.wo_pixel=False
        self.visual_img =False
    def clear(self):
        torch.cuda.empty_cache()
        del self.ad_net
        torch.cuda.empty_cache()

    def fit(self,train_data,**kwargs):
        self.train_data=train_data
        self.train_data_loader= DataLoader(self.train_data, batch_size=kwargs.get("batch_size",64),
                            shuffle=False, num_workers=4,drop_last=False)


            
        self.ad_net.fit_dataset(self.train_data_loader,device=self.device)


    def update_hyper_param(self,**kwargs):
        if kwargs.get("ad_method") == "NNHead":
            self.ad_net.head.set_nn_type(kwargs.get("nn_type"),kwargs.get("k"))
            ld_coefficient=kwargs.get("ld_coefficient",None)
            if ld_coefficient is not None:
                # print(ldn_factor)
                assert( hasattr(self.ad_net.head,"ld_coefficient"))
                self.ad_net.head.ld_coefficient=ld_coefficient
                print("set ld_coefficient to {}".format(ld_coefficient))

    def run_eval(self, test_data, **kwargs):

        self.test_data=test_data
        self.test_data_loader= DataLoader(self.test_data, batch_size=kwargs.get("batch_size",1),
                            shuffle=False, num_workers=4,drop_last=False)

        metrics_result={"train_img":len(self.train_data),"test_data":len(self.test_data)}
        if  hasattr(self.ad_net.head,"ld_coefficient"):
            metrics_result["ld_coefficient"] = self.ad_net.head.ld_coefficient
        self.ad_net.eval()
        metrics_result=self._run_eval(metrics_result)
        if kwargs.get("vis_img",False):

            # self._visual(metrics_result["best_threshold"])
            self._visual(metrics_result["best_threshold"])
            # self._visual(metrics_result["normal_support"])
        print(metrics_result)
        EXPER = ET.get_existing_exp()  # 获取单实例 试验 instance
        EXPER.get("res_csv").append_one_row(metrics_result)

        ###################################

        
    def _run_eval(self,metrics_result):
        device=self.device
        total_time=0
        head_total_time=0 #只统计 model head 时间
        iter_num=0
        img_labels = []
        # pixel_labels =[]
        img_scores = []
        EXPER = ET.get_existing_exp()  # 获取单实例 试验 instance
        if self.visual_Tool is not None:
            save_dir=os.path.join(EXPER.get("res_csv").csv_path[:-4],
                                   EXPER.get("cul_data"))
             
            self.visual_Tool.set_save_dir(save_dir)
        self.metric.reset()#      
        with torch.no_grad():
            print("testing...")
            for batch in tqdm(self.test_data_loader):
                pixel_label=None
                if len(batch)==3:
                    x, img_label,pixel_label=batch
                    x, img_label,pixel_label=x.to(device),img_label.to(device),pixel_label.to(device)
                if len(batch)==2:
                    x, img_label=batch
                    x, img_label=x.to(device),img_label.to(device)
                if len(batch)==4:
                    x, img_label,pixel_label,img_name_batch=batch
                    x, img_label,pixel_label=x.to(device),img_label.to(device),pixel_label.to(device)
                
                start = time.time()
                torch.cuda.synchronize()
                img_score,pixel_score=self.ad_net(x.to(device))
                torch.cuda.synchronize()
                end = time.time()
                total_time+=(end-start)
                head_total_time+=self.ad_net.buff["head_runtime"]
                iter_num+=1
                img_scores.append(img_score)
                img_labels.append(img_label)
                wo_pixel = self.wo_pixel or (pixel_score is  None  or  pixel_label is  None )
                if pixel_label is not None:
                    pixel_label=pixel_label.cpu().numpy().astype(np.int64)
                    # print(pixel_label.shape)
                    # print(np.unique(pixel_label))
                if not wo_pixel :
                    pixel_score = self.anomaly_segmentor.convert_to_segmentation(pixel_score)
                    self.metric.add_batch(pixel_score.flatten(),pixel_label.flatten())

                # print(pixel_score.shape,pixel_label.shape)
            img_labels = torch.cat(img_labels)
            img_scores = torch.cat(img_scores)

            # normal_support=torch.max(img_scores[get_tensor_indice(img_labels,[0])]).item()
            # metrics_result["normal_support"] =normal_support
            average_time=total_time/iter_num
            head_average_time=head_total_time/iter_num

        if not wo_pixel :
            metric_data=self.metric.get()
            pixel_roc_auc=metric_data["roc_auc"]
            anomaly_pixel_roc_auc=metric_data["anomaly_roc_auc"]


        else:
               pixel_roc_auc=0
        self.metric.reset()
        self.metric.add_batch(img_scores.cpu().numpy(),img_labels.cpu().numpy().astype(np.int64))
        img_metric_result=self.metric.get()
        metrics_result["roc_auc"]=img_metric_result["roc_auc"]

        if "roc_auc_class1" in img_metric_result.keys():
            metrics_result["roc_auc_class1"] = img_metric_result["roc_auc_class1"]
        if "roc_auc_class2" in img_metric_result.keys():
            metrics_result["roc_auc_class2"] = img_metric_result["roc_auc_class2"]

        self.class_metric.reset()
        self.class_metric.add_batch(
            img_scores.cpu().numpy(),np.where(img_labels.cpu().numpy().astype(np.int64)>0,1,0))



        if not wo_pixel:
            metrics_result["anomaly_pixel_roc_auc"]=anomaly_pixel_roc_auc
            metrics_result["pixel_roc_auc"]=pixel_roc_auc
        metrics_result["average_time"]=average_time
        metrics_result["head_average_time"]=head_average_time
        metrics_result["iter_num"]=iter_num
        metrics_result["cul_type"]=EXPER.get_param().cul_type
        for k, v in self.class_metric.get().items():
            metrics_result[k]=v
        return  metrics_result

    def _visual(self,threshould):
        device=self.device
        total_time=0
        head_total_time=0 #只统计 model head 时间
        iter_num=0
        img_labels = []
        # pixel_labels =[]
        img_scores = []
        wo_pixel=False
        if self.visual_Tool is not None:
            EXPER = ET.get_existing_exp()  # 获取单实例 试验 instance
            save_dir=os.path.join(EXPER.get("res_csv").csv_path[:-4],
                                   EXPER.get("cul_data"))
             
            self.visual_Tool.set_save_dir(save_dir)
        # pixel_scores=[]
        self.metric.reset()#
        test_data_loader= DataLoader(self.test_data, batch_size=1,
                            shuffle=False, num_workers=4,drop_last=False)
        with torch.no_grad():
            for idx,(x, img_label,pixel_label) in enumerate(test_data_loader):
                x, img_label,pixel_label=x.to(device),img_label.to(device),pixel_label.to(device)

                img_score,pixel_score=self.ad_net(x.to(device))

                assert(pixel_label is not None)
                assert(pixel_score is not None)
                assert(self.visual_Tool is not None)
                pixel_score = self.anomaly_segmentor.convert_to_segmentation(pixel_score)
                pixel_label=pixel_label.cpu().numpy().astype(np.int64)
                img_pred= 1 if img_score>=threshould else 0
                
                name= "{}/{}/{}.png".format(img_label.item(),img_pred,idx)
                if img_pred:
                    self.visual_Tool.do_it(x.cpu(),pixel_label,[pixel_score],[name])
                else:
                    self.visual_Tool.do_it(x.cpu(),pixel_label,[np.zeros_like(pixel_score)],[name])
    
