import sys,os
project_dir=os.path.dirname(__file__)
solution_dir=os.path.dirname(os.path.dirname(project_dir))
sys.path.append(solution_dir)


from  src.Data.anomaly_datasets import  MvtecLoco
from src.Frame.checkpoint import CheckPoint
from src.Frame import ComposeJoint
from src.module import *

import src.Frame.exp_tool as ET
import src.Frame.file_tool as FT
from src.Frame import DatasetWrapper
from src.Data.transforms import RandomTranspose
import src.ADModels as AD
from src.ADModels import NN_Type
from src import  global_param


EXPER=ET.Experiment(save_root=global_param.SAVE_ROOT,
                    project_name=ET.get_folder_name_of_file(__file__),
                    exp_name="111",
                    run_name=None,seed=0 )


ET.get_existing_exp()