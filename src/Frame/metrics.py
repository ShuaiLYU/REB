import numpy as  np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
__all__ = ['classificationMetric',"SegmentationMetric"]

from .base import MetricBase,HookBase




class MetricHook(HookBase):
    
    def __init__(self,dataset,metric,period,batch_size=64):
        
        self.dataset=dataset
        self.metric=metric
        self.period=period
        self.data_loader= DataLoader(self.dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0,drop_last=False)
        
    
    def after_epoch(self):
        
        epoch = self.trainer.epo
        if epoch%self.period!=0:
            return
    
        model.train()
        return self.metric.get()
            
            
    def eval_func(self):
        model= self.trainer.model
        model.eval()
        self.metric.reset()
        for x_batch,y_batch in self.data_loader:
            fx_batch=model(x_batch)
            self.metric.addBatch(fx_batch,y_batch)
        
        

class AucMetric(MetricBase):
          
    def get(self):  
        res={}
        fpr, tpr, _ = roc_curve(np.concatenate(self.labels), np.concatenate(self.scores))
        roc_auc = auc(fpr, tpr)
        # res["fpr"]=fpr
        # res["tpr"]=tpr
        res["roc_auc"]=roc_auc
        return res
        

from sklearn.metrics import f1_score
import numpy as np
class BinaryClassMetric(MetricBase):
    
    def get(self):  
        res={}
        best_f1, best_threshold, error_count, fp, fn,P,N= self.compute_f1_threshold(
            np.concatenate(self.labels),  np.concatenate(self.scores))
        res["best_f1"]=best_f1
        res["best_threshold"]=best_threshold
        res["error_count"]=error_count
        res["fp"]=fp
        res["fn"]=fn
        res["P"]=P
        res["N"]=N
        return res
        
    @staticmethod
    def compute_f1_threshold(labels, scores):
        """
        计算F1 score、最优阈值、错误分类数、FP和FN
        :param labels: 一维数组，元素为0或1，表示真实标签
        :param scores: 一维数组，元素为实数，表示分类器的预测得分
        :return: F1 score, 最优阈值, 错误分类数, FP, FN
        """
        # 计算F1 score和最优阈值
        f1_scores = []
        thresholds = np.unique(scores) # 尝试所有可能的阈值
        # print(thresholds)
        for t in thresholds:
            f1 = f1_score(labels, scores > t)
            f1_scores.append(f1)
        f1_scores = np.array(f1_scores)
        best_f1 = f1_scores.max()
        best_threshold = thresholds[f1_scores.argmax()]

        # 将预测得分转换为预测标签，并计算错误分类数、FP和FN
        predicted_labels = (scores > best_threshold).astype(int)
        error_count = np.sum(predicted_labels != labels)
        fp = np.sum(np.logical_and(predicted_labels == np.ones_like(labels), labels == np.zeros_like(labels)))
        fn = np.sum(np.logical_and(predicted_labels == np.zeros_like(labels), labels == np.ones_like(labels)))
        P,N=np.sum(labels == np.ones_like(labels)),np.sum(labels == np.zeros_like(labels))
        
        return best_f1, best_threshold, error_count, fp, fn,P,N
# compute_f1_threshold([1,0,1,0,1],[0.5,0.2,0.89,0.8,0.4])
        
        
class classificationMetric(MetricBase):

    def __init__(self,numClass):

        self.labels=[]
        self.scores = []
        self.numClass=numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
        print("classificationMetric, cls: {}".format(self.numClass))


    def reset(self):
        self.labels=[]
        self.scores=[]
        self.confusionMatrix=np.zeros((self.numClass, self.numClass))
        return self
    def add_batch(self,scores,labels):
        assert labels.shape==scores.shape[:-1]
        self.labels.append(labels)
        self.scores.append(scores)
        predicts=np.argmax(scores,axis=-1)
        cm=self.genConfusionMatrix(predicts,labels)
        self.confusionMatrix+=cm

        
    def get(self) ->dict:
        res={}
        res["acc"]=self.Accuracy()
        res["meanPrecision"]=self.meanPrecision()
        res["meanRecall"]=self.meanRecall()
        return res

    def genConfusionMatrix(self, Predict, Label):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (Label >= 0) & (Label < self.numClass)
        label = self.numClass * Label[mask] + Predict[mask]
        count = np.bincount(label.astype(np.int), minlength=self.numClass**2)
        cm = count.reshape(self.numClass, self.numClass)
        return cm



    def Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPrecision(self):
        # return each category  accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)

    def meanPrecision(self):
        classPre = self.classPrecision()
        meanPre = np.nanmean(classPre)
        return meanPre

    def classRecall(self):
        return np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
    
    def meanRecall(self):
        class_Recal = self.classRecall()
        mean_recall = np.nanmean(class_Recal)
        return mean_recall
    
    def save_cm_map(self,save_path,axis_name=None,dpi=200):
        fig = plt.figure()
        df=pd.DataFrame(self.confusionMatrix.astype(np.int),index=axis_name,columns=axis_name)
        fig=sns.heatmap(df,annot=True)
        fig.get_figure().savefig(save_path, dpi = dpi)




class SegmentationMetric(MetricBase):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

        
        
    def addBatch(self, imgPredict, imgLabel):
        #print(imgPredict.shape, imgLabel.shape)
        assert imgPredict.shape == imgLabel.shape,"imgPredict shape:{}" "imgLabel shape:{}".format(imgPredict.shape,imgLabel.shape)

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
        
        
    def get(self)->dict:
        res={}
        res["pixelAccuracy"]=self.pixelAccuracy()
        res["meanIntersectionOverUnion"]=self.meanIntersectionOverUnion()
        return res
        
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def clsIntersectionOverUnion(self,cls):
        assert cls<self.numClass
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        return IoU[cls]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


