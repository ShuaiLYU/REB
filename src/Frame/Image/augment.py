import imgaug.augmenters as iaa
import  numpy   as np


"""
amend from  
https://github.com/VitjanZ/DRAEM/blob/main/data_loader.py#L64

"""
from  PIL import  Image
class AugBase(object):


    def get_aug(self) ->iaa.Sequential:
        pass

    def __call__(self,img:Image)->Image:
        # print(np.array(img).size,img)
        array= self.get_aug()(image=np.array(img))
        return Image.fromarray(array)


class RandomAugment(AugBase):

    def __init__(self, num_choice):

        self.num_choice=num_choice

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

    def get_aug(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), self.num_choice, replace=False)
        aug = iaa.Sequential([self.augmenters[i] for i in aug_ind])
        return aug


    class RotateAug(AugBase):

        def __init__(self,angle_range=(-90,90)):
            self.angle_range=angle_range
            self.augmenters = [iaa.Affine(rotate=angle_range)]

        def get_aug(self):
            aug = iaa.Sequential(self.augmenters)
            return aug
