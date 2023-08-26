import os.path

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageDraw
from joblib import Parallel, delayed
import torch
import random
import src
from src.Data.utils import _pair
import numpy as np
from torchvision import transforms

CATE_OBJECT = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor","zipper"]
CATE_TEXTURE = ["carpet", "grid", "leather", "tile", "wood"]


class MVTecAD(Dataset):

    def __init__(self, root_dir, cate_name, size, transform=None,
                 mode="train", suffix=".png", **kwargs):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            subdata (string): defect to load.
            transform: Transform to apply to data
            mode: "train" loads training samples "test" test samples default "train"
        """
        # assert category in ["mvtec","mvtec_loco"]
        # self.category=category
        self.root_dir = Path(root_dir)
        self.subdata = cate_name
        self.transform = transform
        self.mode = mode
        self.imagesize = _pair(size)
        self.scale = 1
        self.pre_load = False
        self.specific_data_len = kwargs.get("specific_data_len", None)
        self.suffix = suffix
        # find test images
        # print(self.root_dir,self.subdata,self.suffix)
        self.load_data()

        default_return_saliency = True if self.subdata in CATE_OBJECT else False
        self.return_saliency = kwargs.get("return_saliency", default_return_saliency)

    def load_data(self):
        if self.mode == "train":
            self.image_names = list((self.root_dir / self.subdata / "train" / "good").glob("*" + self.suffix))
            # print(self.image_names)
        else:
            self.image_names = list(
                (self.root_dir / self.subdata / "test").glob(str(Path("*") / ("*" + self.suffix))))
        if self.mode == "train" and len(self) < 500:
            # no pre-load when img size is bigger than 300
            self.pre_load = True

            print("pre-loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: self.read_img(file)) for file in self.image_names)
            # self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
            self.imgs = Parallel(n_jobs=10)(
                delayed(lambda file: self.read_img(file))(file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")

    def __len__(self):
        data_len = len(self.image_names) if self.specific_data_len is None else self.specific_data_len
        return data_len

    def __getitem__(self, idx):
        # idx=idx%len(self.image_names)
        imagepath = self.image_names[idx]
        # print(imagepath)
        img = None
        if self.pre_load:
            img = self.imgs[idx].copy()
        else:
            img = self.read_img(imagepath)
        # print(img.size,idx)
        label_name = self.imgpath_to_labelname(imagepath)
        label = self.encode_label(label_name)
        saliency = None
        mask = None

        if not hasattr(self, "mask_tran"):
            self.mask_tran = transforms.ToTensor()
        if self.mode == "test":
            if label == 0:

                mask = np.zeros([self.imagesize[1], self.imagesize[0]]).astype(np.uint8)
            else:
                mask = self.imgpath2mask(imagepath).astype(np.uint8)
            mask = self.mask_tran(mask)

        if self.return_saliency and label == 0 and self.mode == "train":
            saliency = self.get_saliency(imagepath)

        if self.transform is not None:
            # print(img,saliency,mask)
            # print(self.transform)
            img, saliency, mask = self.transform([img, saliency, mask])

            # print("img_shpae",np.array(img).shape)
        return {"img": img, "label": label, "label_name": label_name, "imagepath": imagepath, "saliency": saliency,
                "mask": mask}

    def read_img(self, img_path):
        return Image.open(img_path).resize(self.imagesize).convert("RGB")

    def cal_imgsize(self, short_side, stride=32):
        assert (short_side % stride == 0)
        imagepath = self.image_names[0]
        w, h = Image.open(imagepath).size
        factor = short_side / min(w, h)
        new_w, new_h = factor * w, factor * h
        new_w = int((new_w + stride / 2) // stride * stride)
        new_h = int((new_h + stride / 2) // stride * stride)
        return new_w, new_h

    def imgpath_to_labelname(self, path):
        label_name = path.parts[-2]
        return label_name

    def encode_label(self, label_name):
        return 0 if label_name == "good" else 1

    def get_saliency(self, img_path):
        saliency = Image.open(str(img_path).replace(self.suffix, "_saliency.jpg")).resize(self.imagesize)
        return saliency

    def get_wightSampler_wight(self):
        weights = []
        # print(len(self))
        for data in self.image_names:
            w = 1
            weights.append(w)
        return weights


    def imgpath2mask(self, imgpath):
        maskpath = (str(imgpath)).replace("test", "ground_truth").replace(self.suffix, "_mask.png")
        assert (os.path.exists(maskpath)), print(maskpath)
        mask = np.array(Image.open(maskpath).resize(self.imagesize))
        mask[mask > 0] = 255
        return mask


class MvtecLoco(MVTecAD):

    # def imgpath2makpath(self, imgpath):
    #     maskpath = (str(imgpath)).replace("test", "ground_truth").replace(".png", "_mask.png")
    #     if not os.path.exists(maskpath):
    #         maskpath = (str(imgpath)[:-4]).replace("test", "ground_truth") + "/000.png"
    #
    #     assert (os.path.exists(maskpath)), print(maskpath)
    #     return maskpath

    def encode_label(self, label_name):
        labels=["good","structural_anomalies","logical_anomalies"]
        labels.index(label_name)
        return labels.index(label_name)


    def imgpath2mask(self, imgpath):
        mask = np.zeros([self.imagesize[1], self.imagesize[0]])
        for i in range(3):
            maskpath = (str(imgpath)[:-4]).replace("test", "ground_truth") + "/00{}.png".format(i)
            if os.path.exists(maskpath):
                mask += np.array(Image.open(maskpath).resize(self.imagesize)).squeeze()
        mask[mask > 0] = 255
        return mask



import glob
from PIL import Image


class DtdDataset(object):

    def __init__(self,root):
        self.samples = sorted(glob.glob(root + "/images/*/*.jpg"))
        # print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        def read_img(img_path):
            return Image.open(img_path).convert("RGB")

        return {"img": read_img(self.samples[idx])}


# if __name__ == "__main__":

#     DATA_DIR = "I:\DATASETS\mvtec_loco_anomaly_detection.tar\mvtec_loco_anomaly_detection"
#     data=MvtecLoco(DATA_DIR,"juice_bottle",128,mode="test")

#     img=data.imgpath2mask(r"I:\DATASETS\mvtec_loco_anomaly_detection.tar\mvtec_loco_anomaly_detection\juice_bottle\test\logical_anomalies\000.png")
#     img=img.squeeze()
#     import matplotlib.pyplot   as plt

#     plt.figure("Image")  # 图像窗口名称
#     plt.imshow(img)

#     plt.show()
