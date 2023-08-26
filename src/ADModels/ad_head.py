
__all__ =['AdHeadBase','GdeAdHead',"AvgDenseHead",
          "KnnAdHead","CoresetKnnAdHead",
          "LdnKnnAdHead","CoreSetLdnKnnAdHead",
          "CoresetLofAdHead","LofAdHead",
          "CoresetLdofAdHead", "LdofAdHead",
          "KthnnAdHead","CoresetKthnnAdHead","NNHead","NN_Type"]

from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import tqdm
from .utils import ModuleCP
class AdHeadBase(ModuleCP):

    def __init__(self):
        super(AdHeadBase,self).__init__()

    def fit(self, embeddings):
        raise NotImplementedError       
    


class DenseHead(AdHeadBase):
    def __init__(self, last_layer, head_layers, num_classes, **kwargs):
        super(DenseHead, self).__init__()
        self.dense = self.class_mlp_head(last_layer, head_layers, num_classes)

    def forward(self, x):
        x = x.squeeze()
        assert len(x.shape)==2
        return self.dense(x)

    def class_mlp_head(self, last_layer, head_layers, num_classes):
        # create MPL head as seen in the code in: https://github.com/uoguelph-          mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        sequential_layers.append(nn.Linear(last_layer, num_classes))
        return nn.Sequential(*sequential_layers)


class AvgDenseHead(DenseHead):
    

    def __init__(self,last_layer,head_layers,num_classes,**kwargs):
        super(AvgDenseHead,self).__init__(last_layer,head_layers,num_classes,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense=self.class_mlp_head(last_layer,head_layers,num_classes)
    
    
    def forward(self,x):
        # print(x.shape)
        x=x.permute(0,3,1,2)
        # print(x.shape)
        x=self.avgpool(x).squeeze()
        # print(x.shape)
        return self.dense(x)


class MaxpDenseHead(DenseHead):

    def __init__(self, last_layer, head_layers, num_classes, **kwargs):
        super(MaxpDenseHead, self).__init__(last_layer, head_layers, num_classes, **kwargs)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dense = self.class_mlp_head(last_layer, head_layers, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        x = self.maxpool(x).squeeze()
        # print(x.shape)
        return self.dense(x)

class GdeAdHead(AdHeadBase):
    

    def __init__(self,**kwargs):
        super(GdeAdHead,self).__init__()
    
    """

    Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """
    def fit(self, embeddings:torch.Tensor):
        print(embeddings.shape)
        if (len(embeddings.shape)>2):
            embeddings=embeddings.view(-1,embeddings.shape[-1])
        # print(embeddings.shape)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        self.register_buffer("mean",torch.mean(embeddings, axis=0))
        inv_conv=LedoitWolf().fit(embeddings.cpu()).precision_
        self.register_buffer("inv_cov",torch.Tensor(inv_conv).to(embeddings.device))
        
    def forward(self, embeddings):
        if (len(embeddings.shape)>2):
            embeddings=embeddings.view(-1,embeddings.shape[-1])
        # print(embeddings.shape）
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances,None

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()

from .PatchCore.common import NearestNeighbourScorer,FaissNN
from .PatchCore.sampler import get_sampler
    
def anomaly_score( x):
    was_numpy = False
    if isinstance(x, np.ndarray):
        was_numpy = True
        x = torch.from_numpy(x)
    while x.ndim > 1:
        x = torch.max(x, dim=-1).values
    if was_numpy:
        return x.numpy()
    return x

############################################################################################################
from enum import Enum
class NN_Type(Enum):
    knn = 0
    kthnn = 1
    lof = 2
    ldof = 3
    ldknn = 4

class NNHead(AdHeadBase):

    def __init__(self,n_nearest_neighbours=4,use_gpu=True,num_workers=4,coreset_percent=1,**kwargs):
        super(NNHead,self).__init__()
        self._nn_method=FaissNN(use_gpu, num_workers)
        self.n_nearest_neighbours=n_nearest_neighbours
        self._knn_func=None
        self.coreset_percent=coreset_percent
        assert(self.coreset_percent<=1 and self.coreset_percent>0)
         
        self.nn_type=kwargs.get("nn_type",None)
    def fit(self, embeddings):
        if (len(embeddings.shape)>2):
            embeddings=embeddings.reshape(-1,embeddings.shape[-1])
        self.register_buffer("embeddings", torch.Tensor(embeddings))
        if self.coreset_percent<1:
            if not hasattr(self,"sampler"):
                self.sampler=get_sampler("approx_greedy_coreset",self.coreset_percent,embeddings.device)
            embeddings=embeddings.cpu().numpy()
            core_embeddings = self.sampler.run(embeddings)
        else:
            core_embeddings=embeddings
        self.register_buffer("core_embeddings",torch.Tensor(core_embeddings))
        # if self.nn_type is not None:
        #     self.set_nn_type(self.nn_type)

    def set_nn_type(self,nn_type:NN_Type,n_nearest_neighbours:int,**kwargs):
        if hasattr(self,"ld_dist") : del self.ld_dist
        if hasattr(self, "lrd") : del self.lrd
        torch.cuda.empty_cache()
        self.n_nearest_neighbours=n_nearest_neighbours
        self.nn_type=NN_Type(nn_type)
        if self.nn_type ==NN_Type.knn:
            self._nn_method.fit(self.core_embeddings.cpu().numpy())
            self._knn = lambda query: self._nn_method.run(
                self.n_nearest_neighbours, query)
        elif self.nn_type==NN_Type.kthnn:
            self._nn_method.fit(self.core_embeddings.cpu().numpy())
            self._knn = lambda query: self._nn_method.run(
                self.n_nearest_neighbours, query)
        elif self.nn_type==NN_Type.lof:
            self._nn_method.fit(self.core_embeddings.cpu().numpy())
            self._knn = lambda query: self._nn_method.run(
                self.n_nearest_neighbours, query)
            D, I = self._knn(self.core_embeddings.cpu().numpy())
            D = D[:, 1:]  # 去掉第一个距离，即自身距离
            # I = I[:, 1:]  # 去掉第一个索引，即自身索引
            k_dist = D[:, -1]  # k 邻近距离
            reach_dist = np.maximum(D, k_dist.reshape(-1, 1))  # 可达距离
            lrd = 1 / np.mean(reach_dist, axis=1)  # 局部可达密度
            self.register_buffer("lrd", torch.Tensor(lrd))
        elif self.nn_type == NN_Type.ldof:
            self._nn_method.fit(self.core_embeddings.cpu().numpy())
            self._knn = lambda query: self._nn_method.run(
                self.n_nearest_neighbours, query)
        elif self.nn_type == NN_Type.ldknn:
            ld_coefficient=kwargs.get("ld_coefficient",1)
            ld_dist=self.get_ld_dist(self.embeddings.cpu().numpy(),self.core_embeddings.cpu().numpy())
            self.register_buffer("ld_dist", torch.Tensor(ld_dist))
            self.ld_coefficient = ld_coefficient
            self._nn_method.fit(self.core_embeddings.cpu().numpy())
            self._knn = lambda query: self._nn_method.run( 1, query)

    def get_ld_dist(self,features,core_features):
        self._nn_method.fit(features)
        knn=lambda query: self._nn_method.run(self.n_nearest_neighbours+1, query)

        def batched(array, batch_size):
            n_iter = int(np.ceil(len(array) / batch_size))
            def get_batch(i):
                left, right = i * batch_size, min((i + 1) * batch_size, len(array))
                return  array[left:right, :]
            return n_iter,get_batch

        n_iter, get_batch=batched(core_features,1024)
        reach_distances=[]

        print("get_ld_dist ....")
        for  idx in tqdm.tqdm(range(n_iter)):
            batch=get_batch(idx)
            query_distances, _= knn(batch)
            mask = query_distances[:, 0] == 0
            dist = np.where(mask[:, None], query_distances[:, 1:self.n_nearest_neighbours+1],
                         query_distances[:, 0:self.n_nearest_neighbours])
            reach_distances.append(np.array(dist))
        torch.cuda.empty_cache()
        reach_distances= np.concatenate(reach_distances,axis=0)
        del  knn
        return reach_distances



    def forward(self, embeddings, return_patch_res=True):
        assert (hasattr(self,"_knn"))
        spatial_shape = embeddings.shape[:-1]
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        anomaly_scores=self._forward(embeddings)
        patch_anomaly_scores = anomaly_scores.reshape(*spatial_shape)
        image_anomaly_scores = F.adaptive_max_pool2d(F.avg_pool2d(patch_anomaly_scores, (3, 3)), (1, 1)).view(-1, 1)
        if return_patch_res:
            return image_anomaly_scores, patch_anomaly_scores
        else:
            return image_anomaly_scores

    def _forward(self, embeddings):
        num_embed=embeddings.shape[0]
        query_distances, query_nns = self._knn(embeddings.cpu().numpy())
        query_distances,query_nns=query_distances.reshape(num_embed,-1), query_nns.reshape(num_embed,-1)
        if self.nn_type==NN_Type.knn:
            anomaly_scores = torch.Tensor(np.mean(query_distances, axis=-1))
            return anomaly_scores
        elif  self.nn_type==NN_Type.kthnn:
            anomaly_scores = torch.Tensor(query_distances[:, -1])
            return anomaly_scores
        elif self.nn_type==NN_Type.lof:
            assert (hasattr(self,"lrd"))
            D, I = query_distances, query_nns
            k_dist = D[:, -1]  # k 邻近距离
            reach_dist = np.maximum(D, k_dist.reshape(-1, 1))  # 可达距离
            lrd = 1 / np.mean(reach_dist, axis=1)  # 局部可达密度
            neighbor_lrd=self.lrd[I]
            lof = torch.mean(neighbor_lrd, dim=1) / torch.Tensor(lrd)  # 局部离群因子
            return lof
        elif self.nn_type==NN_Type.ldof:
            D, I = query_distances, query_nns
            knn = self.core_embeddings[I]
            knn_inter_dist = torch.cdist(knn, knn, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')
            knn_inter_dist = torch.sum(knn_inter_dist, dim=[1, 2]) / (
                        self.n_nearest_neighbours * (self.n_nearest_neighbours - 1))
            knn_avg_dist = torch.mean(torch.sqrt(torch.tensor(D)), dim=1).to(knn_inter_dist.device)
            anomaly_scores = knn_avg_dist / knn_inter_dist
            return anomaly_scores
        elif self.nn_type==NN_Type.ldknn:
            assert (hasattr(self,"ld_coefficient"))
            assert (hasattr(self, "ld_dist"))
            ld_dist = torch.mean(self.ld_dist[query_nns].squeeze(), axis=-1, keepdims=True)
            anomaly_scores = torch.Tensor(query_distances) - ld_dist * self.ld_coefficient
            return anomaly_scores



class KnnAdHead(AdHeadBase):

    def __init__(self,n_nearest_neighbours=4,nn_method=FaissNN(True, 4),**kwargs):
        super(KnnAdHead,self).__init__()
        self._nn_method=nn_method
        self.n_nearest_neighbours=n_nearest_neighbours
        self._knn=None

    def fit(self, embeddings):
        if (len(embeddings.shape)>2):
            embeddings=embeddings.reshape(-1,embeddings.shape[-1])
        self.register_buffer("embeddings",torch.Tensor(embeddings))
        self._fit()
        
    def _fit(self):
        self._nn_method.fit(self.embeddings.cpu().numpy())
        self._knn=lambda query: self._nn_method.run(
            self.n_nearest_neighbours, query)
            
    def forward(self, embeddings,return_patch_res=True):
        if self._knn is None: self._fit() 
        spatial_shape=embeddings.shape[:-1]
        if (len(embeddings.shape)>2):
            embeddings=embeddings.reshape(-1,embeddings.shape[-1])
        query_distances, query_nns=self._knn(embeddings.cpu().numpy())
        anomaly_scores = torch.Tensor(np.mean(query_distances, axis=-1))
        patch_anomaly_scores=anomaly_scores.reshape(*spatial_shape)
        image_anomaly_scores=F.adaptive_max_pool2d(F.avg_pool2d(patch_anomaly_scores,(3,3)),(1,1)).view(-1,1)
        # image_anomaly_scores=F.adaptive_max_pool2d(patch_anomaly_scores,(1,1)).squeeze()
        # image_anomaly_scores=anomaly_score(patch_anomaly_scores)
        # print(patch_anomaly_scores.shape,image_anomaly_scores.shape)
        
        if  return_patch_res:
            return  image_anomaly_scores,patch_anomaly_scores 
        else:
            return  image_anomaly_scores





class CoresetKnnAdHead(KnnAdHead):
    
    def __init__(self,coreset_percent,device,n_nearest_neighbours=1,nn_method=FaissNN(True, 4),**kwargs):
        super(CoresetKnnAdHead,self).__init__(n_nearest_neighbours,nn_method)
        self.sampler=get_sampler("approx_greedy_coreset",coreset_percent,device)
            
            
    def fit(self, embeddings):
        if (len(embeddings.shape)>2):
            embeddings=embeddings.reshape(-1,embeddings.shape[-1])
        embeddings=embeddings.cpu().numpy()
        core_embeddings = self.sampler.run(embeddings)
        self.register_buffer("embeddings",torch.Tensor(core_embeddings))
        self._fit()


############################################################################################################
class KthnnAdHead(AdHeadBase):

    def __init__(self, n_nearest_neighbours=4, nn_method=FaissNN(True, 4),**kwargs):
        super(KthnnAdHead, self).__init__()
        self._nn_method = nn_method
        self.n_nearest_neighbours = n_nearest_neighbours
        self._knn = None

    def fit(self, embeddings):
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        self.register_buffer("embeddings", torch.Tensor(embeddings))
        self._fit()

    def _fit(self):
        self._nn_method.fit(self.embeddings.cpu().numpy())
        self._knn = lambda query: self._nn_method.run(
            self.n_nearest_neighbours, query)

    def forward(self, embeddings, return_patch_res=True):
        if self._knn is None: self._fit()
        spatial_shape = embeddings.shape[:-1]
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        query_distances, query_nns = self._knn(embeddings.cpu().numpy())

        anomaly_scores = torch.Tensor(query_distances[:,-1]).view(-1,1)
        patch_anomaly_scores = anomaly_scores.reshape(*spatial_shape)
        image_anomaly_scores = F.adaptive_max_pool2d(F.avg_pool2d(patch_anomaly_scores, (3, 3)), (1, 1)).view(-1,1)
        # image_anomaly_scores=F.adaptive_max_pool2d(patch_anomaly_scores,(1,1)).squeeze()
        # image_anomaly_scores=anomaly_score(patch_anomaly_scores)
        # print(patch_anomaly_scores.shape,image_anomaly_scores.shape)

        if return_patch_res:
            return image_anomaly_scores, patch_anomaly_scores
        else:
            return image_anomaly_scores


class CoresetKthnnAdHead(KnnAdHead):

    def __init__(self, coreset_percent, device, n_nearest_neighbours=1, nn_method=FaissNN(True, 4),**kwargs):
        super(CoresetKthnnAdHead, self).__init__(n_nearest_neighbours, nn_method)
        self.sampler = get_sampler("approx_greedy_coreset", coreset_percent, device)

    def fit(self, embeddings):
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        embeddings = embeddings.cpu().numpy()
        core_embeddings = self.sampler.run(embeddings)
        self.register_buffer("embeddings", torch.Tensor(core_embeddings))
        self._fit()


############################################################################################################
class LofAdHead(AdHeadBase):

    def __init__(self, n_nearest_neighbours=4, nn_method=FaissNN(True, 4),**kwargs):
        super(LofAdHead, self).__init__()
        self._nn_method = nn_method
        self.n_nearest_neighbours = n_nearest_neighbours
        self._knn = None

    def fit(self, embeddings):

        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        self.register_buffer("embeddings", torch.Tensor(embeddings))
        self._fit()

    def _fit(self):
        print("LofAdHead . fiting...")
        self._nn_method.fit(self.embeddings.cpu().numpy())
        self._knn = lambda query: self._nn_method.run(
            self.n_nearest_neighbours, query)
        D, I = self._knn(self.embeddings.cpu().numpy())
        D = D[:, 1:]  # 去掉第一个距离，即自身距离
        I = I[:, 1:]  # 去掉第一个索引，即自身索引
        k_dist = D[:, -1]  # k 邻近距离
        reach_dist = np.maximum(D, k_dist.reshape(-1, 1))  # 可达距离
        self.lrd =1 / np.mean(reach_dist, axis=1)  # 局部可达密度

    print("LofAdHead  fit finish...")
    def forward(self, embeddings, return_patch_res=True):
        if self._knn is None: self._fit()
        spatial_shape = embeddings.shape[:-1]
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        D, I = self._knn(embeddings.cpu().numpy())
        k_dist = D[:, -1]  # k 邻近距离
        reach_dist = np.maximum(D, k_dist.reshape(-1, 1))  # 可达距离
        lrd = 1 / np.mean(reach_dist, axis=1)  # 局部可达密度
        # lof = np.sum(self.lrd[I] / lrd.reshape(-1, 1), axis=1) / k # 局部离群因子
        lof = np.mean(self.lrd[I], axis=1) / lrd  # 局部离群因子

        anomaly_scores = torch.Tensor(lof)
        patch_anomaly_scores = anomaly_scores.reshape(*spatial_shape)
        image_anomaly_scores = F.adaptive_max_pool2d(F.avg_pool2d(patch_anomaly_scores, (3, 3)), (1, 1)).view(-1,1)

        if return_patch_res:
            return image_anomaly_scores, patch_anomaly_scores
        else:
            return image_anomaly_scores


class CoresetLofAdHead(LofAdHead):

    def __init__(self, coreset_percent, device, n_nearest_neighbours=1, nn_method=FaissNN(True, 4),**kwargs):
        super(CoresetLofAdHead, self).__init__(n_nearest_neighbours, nn_method)
        self.sampler = get_sampler("approx_greedy_coreset", coreset_percent, device)

    def fit(self, embeddings):
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        embeddings = embeddings.cpu().numpy()
        core_embeddings = self.sampler.run(embeddings)
        self.register_buffer("embeddings", torch.Tensor(core_embeddings))
        self._fit()
############################################################################################################
class LdofAdHead(AdHeadBase):

    def __init__(self, n_nearest_neighbours=4, nn_method=FaissNN(True, 4),**kwargs):
        super(LdofAdHead, self).__init__()
        self._nn_method = nn_method
        self.n_nearest_neighbours = n_nearest_neighbours
        self._knn = None

    def fit(self, embeddings):

        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        self.register_buffer("embeddings", torch.Tensor(embeddings))
        self._fit()

    def _fit(self):
        print("LdofAdHead . fiting...")
        self._nn_method.fit(self.embeddings.cpu().numpy())
        self._knn = lambda query: self._nn_method.run(
            self.n_nearest_neighbours, query)

    def forward(self, embeddings, return_patch_res=True):
        if self._knn is None: self._fit()
        spatial_shape = embeddings.shape[:-1]
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        D, I = self._knn(embeddings.cpu().numpy())
        knn=self.embeddings[I]
        knn_inter_dist = torch.cdist(knn, knn, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary')
        knn_inter_dist=torch.sum(knn_inter_dist,dim=[1,2])/(self.n_nearest_neighbours*(self.n_nearest_neighbours-1))
        knn_avg_dist=torch.mean(torch.sqrt(torch.tensor(D)),dim=1).to(knn_inter_dist.device)
        anomaly_scores = knn_avg_dist/knn_inter_dist
        patch_anomaly_scores = anomaly_scores.reshape(*spatial_shape)
        image_anomaly_scores = F.adaptive_max_pool2d(F.avg_pool2d(patch_anomaly_scores, (3, 3)), (1, 1)).view(-1,1)
        if return_patch_res:
            return image_anomaly_scores, patch_anomaly_scores
        else:
            return image_anomaly_scores
class CoresetLdofAdHead(LdofAdHead):

    def __init__(self, coreset_percent, device, n_nearest_neighbours=1, nn_method=FaissNN(True, 4),**kwargs):
        super(CoresetLdofAdHead, self).__init__(n_nearest_neighbours, nn_method)
        self.sampler = get_sampler("approx_greedy_coreset", coreset_percent, device)

    def fit(self, embeddings):
        if (len(embeddings.shape) > 2):
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        embeddings = embeddings.cpu().numpy()
        core_embeddings = self.sampler.run(embeddings)
        self.register_buffer("embeddings", torch.Tensor(core_embeddings))
        self._fit()
############################################################################################################

class LdnKnnAdHead(AdHeadBase):
    
    
    def __init__(self,n_nearest_neighbours=1,n_reach_field=10, ldn_factor=1,nn_method=FaissNN(True, 4),**kwargs):
        super(LdnKnnAdHead,self).__init__()
        self.n_nearest_neighbours=n_nearest_neighbours
        self.n_reach_field=n_reach_field
        self.ldn_factor=ldn_factor
        self._nn_method=nn_method
        
        
        
    def _fit(self):
        self._nn_method.fit(self.embeddings.cpu().numpy())
        self._knn=lambda query: self._nn_method.run(
            self.n_nearest_neighbours, query)
        
    def fit(self, embeddings):
        if (len(embeddings.shape)>2):
            embeddings=embeddings.view(-1,embeddings.shape[-1])
        embeddings=embeddings.cpu().numpy()
        coreset_embeddings=copy.deepcopy(embeddings)
        if self.ldn_factor>0:
            self.register_buffer("_reach_distances",torch.Tensor(self.get_reach_distances(embeddings,coreset_embeddings)))
        self.register_buffer("embeddings",torch.Tensor(coreset_embeddings))
        self._fit()
        
    def forward(self, embeddings,return_patch_res=True):
        if self._knn is None: self._fit()
        spatial_shape=embeddings.shape[:-1]
        if (len(embeddings.shape)>2):
            embeddings=embeddings.view(-1,embeddings.shape[-1])
        embeddings=embeddings.cpu().numpy()
        
        query_distances, query_nns=self._knn(embeddings)
        
        if self.ldn_factor>0:
            reach_distance = torch.mean(self._reach_distances[query_nns].squeeze(), axis=-1,keepdims=True)
            query_distances = torch.Tensor(query_distances) - reach_distance*self.ldn_factor
        # print(self.ldn_factor)
        anomaly_scores = query_distances.mean(axis=-1)


        patch_anomaly_scores=torch.Tensor(anomaly_scores).reshape(*spatial_shape)
        # print(patch_anomaly_scores.shape)
        # image_anomaly_scores=F.adaptive_max_pool2d(patch_anomaly_scores,(1,1)).squeeze()
        image_anomaly_scores=F.adaptive_max_pool2d(F.avg_pool2d(patch_anomaly_scores,(3,3)),(1,1)).view(-1,1)
        if  return_patch_res:
            return  image_anomaly_scores,patch_anomaly_scores 
        else:
            return  image_anomaly_scores


    # def get_reach_distances(self,features,core_features):
    #     self._nn_method.fit(features)
    #     knn=lambda query: self._nn_method.run(self.n_reach_field, query)
    #
    #     def batched(array, batch_size):
    #         n_iter = int(np.ceil(len(array) / batch_size))
    #         for i in range(n_iter):
    #             left, right = i * batch_size, min((i + 1) * batch_size, len(array))
    #             yield array[left:right, :]
    #     reach_distances=[]
    #     nns=[]
    #     print("get_reach_distance ....")
    #     for  batch in tqdm.tqdm(batched(core_features,64)):
    #         query_distances, query_nns= knn(batch)
    #         # print("score",score)
    #         reach_distances.append(np.array(query_distances))
    #         nns.append(np.array(query_nns))
    #     torch.cuda.empty_cache()
    #     reach_distances= np.concatenate(reach_distances,axis=0)
    #     nns = np.concatenate(nns, axis=0)
    #     del  knn
    #     return reach_distances

    def get_reach_distances(self,features,core_features):
        self._nn_method.fit(features)
        knn=lambda query: self._nn_method.run(self.n_reach_field+1, query)

        def batched(array, batch_size):
            n_iter = int(np.ceil(len(array) / batch_size))
            def get_batch(i):
                left, right = i * batch_size, min((i + 1) * batch_size, len(array))
                return  array[left:right, :]
            return n_iter,get_batch

        n_iter, get_batch=batched(core_features,1024)
        reach_distances=[]

        print("get_reach_distance ....")
        for  idx in tqdm.tqdm(range(n_iter)):
            batch=get_batch(idx)
            query_distances, _= knn(batch)
            mask = query_distances[:, 0] == 0
            dist = np.where(mask[:, None], query_distances[:, 1:self.n_reach_field+1],
                         query_distances[:, 0:self.n_reach_field])
            reach_distances.append(np.array(dist))
        torch.cuda.empty_cache()
        reach_distances= np.concatenate(reach_distances,axis=0)
        del  knn
        return reach_distances

    
class CoreSetLdnKnnAdHead(LdnKnnAdHead):
    
    def __init__(self,coreset_percent,device,n_nearest_neighbours=1,n_reach_field=10,
                 ldn_factor=1,nn_method=FaissNN(True, 4),**kwargs):
        super(CoreSetLdnKnnAdHead,self).__init__(n_nearest_neighbours,n_reach_field,ldn_factor,nn_method)
        self.sampler=get_sampler("approx_greedy_coreset",coreset_percent,device)
        
    def fit(self, embeddings):
        if (len(embeddings.shape)>2):
            embeddings=embeddings.view(-1,embeddings.shape[-1])
        embeddings=embeddings.cpu().numpy()
        core_embeddings = self.sampler.run(embeddings)
                
        if self.ldn_factor>0:
            self.register_buffer("_reach_distances",\
                                 torch.Tensor(self.get_reach_distances(embeddings,core_embeddings)))
        self.register_buffer("embeddings",torch.Tensor(core_embeddings))
        self._fit()