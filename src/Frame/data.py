


from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import  math
from random import  shuffle

try:
    from collections import Iterable

except ImportError:
    from collections.abc import Iterable


class unifiedBatchSampler(object):
    def __init__(self, data_labels,batch_size):

        super(unifiedBatchSampler,self).__init__()
        self.batch_size=batch_size
        self.cls_dict=self.get_cls_dict(data_labels)
        maxNumCls = max([len(val) for _, val in self.cls_dict.items()])
        self.numBatchCls = math.ceil(maxNumCls / float(self.batch_size))
        self.numCls = len(self.cls_dict)
        self.num = self.numCls * self.numBatchCls * self.batch_size

    def get_cls_dict(self,label_list):
        cls_dict = {}
        for idx,label in enumerate(label_list):
            if label not in cls_dict.keys(): cls_dict[label]=[]
            cls_dict[label].append(idx)
        return cls_dict


    def __len__(self):

        return self.num

    def __iter__(self):

        for b in range(self.numBatchCls):
            for cls in range(self.numCls):
                batch = []
                for i in range(self.batch_size):
                    loc=b*self.batch_size+i
                    loc=loc%len(self.cls_dict[cls])
                    if loc==0:
                        shuffle(    self.cls_dict[cls])
                        # print(self.cls_dict[2])
                    yield self.cls_dict[cls][loc]
                    # batch.append(self.cls_dict[cls][loc])

                # yield batch








from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    
    def __init__(self,dataset, keys=None,new_length=None,transform=None):
        
        self.dataset=dataset
        self.keys=keys
        self.new_length=new_length
        self.transform=transform
        
        if self.new_length is not None:
            self._iidxes=[ i%len(self.dataset) for i  in range(self.new_length)]
        else:
            self._iidxes=[ i%len(self.dataset) for i  in range(len(self.dataset))]
            
    def __len__(self):
        return len(self._iidxes)
    
    def __getitem__(self, idx):
        idx=self._iidxes[idx]
        item=self.dataset[idx]
        if self.transform is not None:
            item["img"]=self.transform(item["img"])
        if self.keys==None:
            pass
        elif  isinstance(self.keys,tuple) or isinstance(self.keys,list):
            item= [item[k] for k in self.keys]
        else:
            item=item[self.keys]
        return item

class DataListWrapper(Dataset):

    def __init__(self,datasets, keys=None,transform=None):
        self.datasets = datasets
        self.keys = keys
        self.transform = transform
        _iidxes=[]

        for i,dataset in enumerate(self.datasets):
            for j in range(len(dataset)):
                _iidxes.append((i,j))
        self._iidxes=_iidxes
        self.new_length=len(_iidxes)

    def __len__(self):
        return len(self._iidxes)

    def __getitem__(self, idx):
        idx1,idx2=self._iidxes[idx]
        item=self.datasets[idx1][idx2]
        if self.transform is not None:
            item["img"]=self.transform(item["img"])
        if self.keys==None:
            pass
        elif  isinstance(self.keys,tuple) or isinstance(self.keys,list):
            item= [item[k] for k in self.keys]
        else:
            item=item[self.keys]
        return item


class ComposeJoint(object):
    def __init__(self, transforms:list=None):
        if not transforms:
            transforms=[]
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = self._iterate_transforms(transform, x)

        return x
    

    def append(self, transform):
        self.transforms.append(transform) 
        
    def _iterate_transforms(self, transforms, x):
        if isinstance(transforms, Iterable):
            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:

            if transforms is not None:
                x = transforms(x)

        return x
    
    