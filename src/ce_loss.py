


import torch
import torch.nn as nn






class SigmoidAndLoss(nn.Module):

    def __init__(self, loss_method,device):
        super(SigmoidAndLoss, self).__init__()
        self.loss_method=loss_method
        self.device=device
        assert(self.loss_method in ["ce","mse","mae"])


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        if self.loss_method == "ce":
            return self.ce_func(x,labels)
        elif self.loss_method=="mse":
            
            return self.mse_func(x,labels)
        # return loss.mean()


    def ce_func(self,x,labels):
        batch_size = x.size(0)
        labels=torch.nn.functional.one_hot(labels).float().to(self.device)

        x=torch.sigmoid(x).clamp(1e-6,1-(1e-6))
        loss=-labels*torch.log(x)-(1-labels)*(torch.log(1-x))
        return loss.mean()

    def mse_func(self,x,labels):
        
        if not hasattr(self,"mse") or self.mse is None:
            self.mse=nn.MSELoss()
        batch_size = x.size(0)
        labels=torch.nn.functional.one_hot(labels).float().to(self.device)
        
        return self.mse(x,labels)


    

        
        
        

class SigCELoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self,num_classes=-1,use_gpu=True,**wargs):
        super(SigCELoss, self).__init__()
        self.num_classes=num_classes
        
        self.class_balance=wargs.get("class_balance",False)

        self.label_one_hot=wargs.get("label_one_hot",True)
        self.weights=wargs.get("weights",None)  
        
    def forward_binary(self,x,labels):
        labels=labels.flatten()
        x=x.flatten()        
        x=torch.sigmoid(x).clamp(1e-6,1-(1e-6))
        loss_pos=-labels*torch.log(x)
        loss_neg=-(1-labels)*(torch.log(1-x))
        loss_pos=torch.sum(loss_pos)/(torch.sum(labels))
        loss_neg=torch.sum(loss_neg)/(torch.sum(1-labels))
        return loss_neg+loss_pos
                   
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # print(x.shape,labels.shape)
        batch_size = x.size(0)
        
        if self.num_classes==1:
            return self.forward_binary(x,labels)
        x=x.reshape(-1,x.shape[-1])
        if self.label_one_hot:
            labels=labels.flatten().to(torch.int64)
            labels=torch.nn.functional.one_hot(labels,num_classes=self.num_classes)
        # print(x.shape,labels.shape)
        x=torch.sigmoid(x).clamp(1e-6,1-(1e-6))
        # print(x.shape,labels.shape)
        # print(x.shape,labels.shape)
        if not self.class_balance:
            loss=-labels*torch.log(x)-(1-labels)*(torch.log(1-x))
            return loss.mean()
        else:
            loss_pos=-torch.sum(labels*torch.log(x))/(torch.sum(labels))
            loss_neg=-torch.sum((1-labels)*(torch.log(1-x)))/(torch.sum(1-labels))
            return loss_pos+loss_neg
    
    

# class SacLoss(SigCELoss):

#     def __init__(self,ks,weights,num_classes=-1,use_gpu=True,**wargs):
#         self.keys=ks
#         self.weights=weights
#         super(SacLoss,self).__init__(num_classes,use_gpu,**wargs)
            
         
            
#     def forward(self,xs,labels):
#         assert(len(xs)==len(labels) and len(xs)==len(self.weights))
#         loss_dict={}
#         for x , label ,w, key in zip (xs,labels,self.weights,self.keys):
#             loss_dict[key]= SigCELoss.forward(self,x,label)*w
#         return loss_dict
 
    
    
                           
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimClrLoss(nn.Module):
    

    def __init__(self,device,n_views,temperature=1,**kwargs):
        super(SimClrLoss,self).__init__()
        self.device=device
        self.n_views=n_views
        self.temperature=temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        self.sphere_loss=kwargs.get("sphere_loss",False)        
        if self.sphere_loss:
            print("using sphere_loss.....")
            
        self.label_mask=None
        
        
    def forward(self, x,y):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        if not self.sphere_loss:
            logits,labels=self.info_nce_loss(x,batch_size)
            x=torch.sigmoid(logits).clamp(1e-6,1-(1e-6))
            loss=-labels*torch.log(x)-(1-labels)*(torch.log(1-x))
            return loss.mean()
        else:
            logits,labels=self.info_nce_loss2(x,batch_size)
            x=torch.sigmoid(logits).clamp(1e-6,1-(1e-6))
       
            pos_loss=-torch.log(x[labels==1])
    
            neg_loss=-torch.log(1-x[labels==0])
            # print(pos_loss.shape,neg_loss.shape)
            return torch.concat([pos_loss,neg_loss]).mean()
    
    
    
    def info_nce_loss(self,features,batch_size):

        # labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = torch.arange(self.n_views)
        # labels=torch.where(labels>0,1,0)
        labels=labels.repeat_interleave(batch_size//self.n_views)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        logits = similarity_matrix / self.temperature
        return logits, labels
    
    
#     def info_nce_loss1(self,features,batch_size):

#         labels = torch.arange(self.n_views)
#         labels=torch.where(labels>0,1,0).repeat_interleave(batch_size//self.n_views) 
 
#         label_mask= ((labels.unsqueeze(0) + labels.unsqueeze(1)==0)).float()
    
#         ignore= ((labels.unsqueeze(0) + labels.unsqueeze(1)>1)).bool()
#         label_mask[ignore]=-1
        
#         ignore = torch.eye(labels.shape[0], dtype=torch.bool)
#         label_mask[ignore]=-1
#         features = F.normalize(features, dim=1)
#         similarity_matrix = torch.matmul(features, features.T)

#         return similarity_matrix, label_mask
    
    
    
    
    def get_mask(self,n_views,batch_size):
        def get_neg_pair_mask(n_views,batch_size):
            labels_1D = torch.arange(batch_size//n_views).repeat(n_views) 
            masks = (labels_1D.unsqueeze(0) == labels_1D.unsqueeze(1)).bool()
            masks[batch_size//n_views:,batch_size//n_views:]=False
            masks[:batch_size//n_views,:batch_size//n_views:]=False
            return masks

        def get_pos_pair_mask(n_views,batch_size):
            labels_1D = torch.arange(n_views).repeat_interleave(batch_size//n_views)     
            masks = (labels_1D.unsqueeze(0) + labels_1D.unsqueeze(1)==0).bool()
            return masks       
        
        masks=torch.ones((batch_size,batch_size)).float()*(-1)
        masks[get_pos_pair_mask(n_views,batch_size)]=1
        masks[get_neg_pair_mask(n_views,batch_size)]=0
        ignore = torch.eye(batch_size,dtype=torch.bool)
        masks[ignore]=-1
        return masks
    
    
    
    def info_nce_loss1(self,features,batch_size):
        
        if self.label_mask is None or self.label_mask.size(0)!=batch_size:    
            self.label_mask=self.get_mask(self.n_views,batch_size).to(self.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T).to(self.device)
        return similarity_matrix, self.label_mask
    
    
    def info_nce_loss2(self,features,batch_size):
        def get_torch_topk_1d(tensor,k,descending=True):
            assert(len(tensor.shape)==1)
            # _,indices = torch.sort(negative_distances,descending=True)
            # topk=(negative_distances)[indices[:k]]
            topk=torch.sort(negative_distances,descending=descending)[0][:k]
            return topk
        labels = torch.arange(self.n_views).repeat_interleave(batch_size//self.n_views) 
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T).to(self.device)
        all_pairs = torch.tensor(list(combinations(range(len(labels)), 2)))
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]])]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]])]
        positive_distances = similarity_matrix[positive_pairs[:, 0], positive_pairs[:, 1]]
        negative_distances = similarity_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        top_positives=get_torch_topk_1d(positive_distances,batch_size)
        bottom_positives=get_torch_topk_1d(positive_distances,batch_size,False)
        top_negatives=get_torch_topk_1d(negative_distances,batch_size)
        bottom_negatives=get_torch_topk_1d(negative_distances,batch_size,False)
        similarity=torch.cat([top_positives,bottom_positives,top_negatives,bottom_negatives])
        labels=torch.cat([torch.ones_like(top_positives).float(),torch.ones_like(bottom_positives).float(),
                          torch.zeros_like(top_negatives).float(),torch.zeros_like(top_negatives).float()])
        # print(similarity.shape,labels.shape)
        return similarity,labels
    
    # def info_nce_loss3(self,features,batch_size):
#         def get_torch_topk_1d(tensor,k):
#             assert(len(tensor.shape)==1)
#             # _,indices = torch.sort(negative_distances,descending=True)
#             # topk=(negative_distances)[indices[:k]]
#             topk=torch.sort(negative_distances,descending=True)[0][:k]
#             return topk
#         labels = torch.arange(self.n_views).repeat_interleave(batch_size//self.n_views) 
#         features = F.normalize(features, dim=1)
#         similarity_matrix = torch.matmul(features, features.T).to(self.device)
#         all_pairs = torch.tensor(list(combinations(range(len(labels)), 2)))
#         positive_pairs = all_pairs[((labels[all_pairs[:, 0]] + labels[all_pairs[:, 1]])==0)  ]
#         negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]])]
#         positive_distances = similarity_matrix[positive_pairs[:, 0], positive_pairs[:, 1]]
#         negative_distances = similarity_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
#         top_positives=get_torch_topk_1d(positive_distances,batch_size)
#         top_negatives=get_torch_topk_1d(negative_distances,batch_size)
#         similarity=torch.cat([top_positives,top_negatives])
#         labels=torch.cat([torch.ones_like(top_positives).float(),torch.zeros_like(top_negatives).float()])
#         # print(similarity.shape,labels.shape)
#         return similarity,labels
       
        
        

    
    
    
    
class LossUnion(SimClrLoss):
    
    def __init__(self,device,n_views,temperature=1):
        nn.Module.__init__(self)
        self.device=device
        self.n_views=n_views
        self.temperature=temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        self.class_loss=SigCELoss((self.device=="cuda"))
        
        
        
    def forward(self,x,**wkargs):
        batch_size = x.size(0)
        loss_crl=SimClrLoss.forward(self,x)
        
        labels = torch.arange(self.n_views).to(self.device)
        # labels=torch.where(labels>0,1,0)
        labels=labels.repeat_interleave(batch_size//self.n_views)
        loss_cls=self.class_loss(x,labels)
        
        return loss_cls+loss_crl
        
        
        

    
        
        
        
        
        