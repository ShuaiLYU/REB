
import src.Frame.exp_tool as ET
from src.ADModels import NN_Type



dtd_data_root="/media/lyushuai/Data/DATASET/dtd"

DATA_DIR = "/home/lyushuai/Datasets/mvtec_anomaly_detection_saliency"
all_types = [
    'screw','cable','capsule','carpet','grid',
    'hazelnut','leather','metal_nut','pill','tile',
    'toothbrush','transistor','wood','bottle','zipper']




Mvtec=ET.Param(data_dir=DATA_DIR,all_types=all_types)

DATA_DIR="/media/lyushuai/Data/DATASET/mvtec_loco_anomaly_detection"
all_types = ['pushpins','juice_bottle','breakfast_box','screw_bag',"splicing_connectors"]
MvtecLOCO=ET.Param(data_dir=DATA_DIR,all_types=all_types)

##################################################

SAVE_ROOT="../OUTPUT"


def get_img_size(data_type):
    return (256, 256)

weight_choices =["last" ,"ImageNet"]




global_param=ET.Param(pretrained=True)
global_param.img_norm=ET.Param(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
global_param.layer="L23"
global_param.coreset=1
global_param.ldknn_factor=1

global_param.ad_method="NNHead"
global_param.nn_types = [ NN_Type.ldknn]

global_param.ks=[3,5,7,9,11]  # for mvtec
# global_param.ks=[39,45,51,57] # for mvtec loco

