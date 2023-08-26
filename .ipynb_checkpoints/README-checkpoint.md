# REB
official source code for REB：Reducing Biases in Representation for Industrial Anomaly Detection  


##
![](pictures/reb.png)






<details open>
<summary>Install</summary>


```bash
$ git clone https://github.com/ShuaiLYU/REB
$ cd REB
$ pip install -r requirements.txt
```

</details>


<details open>
<summary>Training on Mvtec AD </summary>

Run commands below to reproduce results
on [Mvtec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

1. you are advised to download Mvtec ad dataset with saliency from [BaiduNetdisk (code: 1234)](https://pan.baidu.com/s/17w4pUWYqzMUs2FSz8vVWKw) or [OneDrive](https://connectpolyu-my.sharepoint.com/personal/21062579r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F21062579r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2Fpublic%5Fshared%2Fmvtec%5Fanomaly%5Fdetection%5Fsaliency%2Ezip&parent=%2Fpersonal%2F21062579r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2Fpublic%5Fshared&ga=1).  
    you can also generate saliency by yourself with teh [EDN saliency model](https://github.com/yuhuan-wu/EDN)
    
2. modify the dataset path and OUTPUT path  in [global_param.py](src/global_param.py) according to your personal config.  
      the best K for Mvtec is 9. set K value in argparse or [global_param.py](src/global_param.py)
    
3. Run commands below to train
```bash
$ cd  REB/projects/reb_mvtec
#  Self-supervied learning  (fine-turning ImageNet-pretrained model with DefectMaker))  Resnet18
python main.py  -exp_name res18_dm_com6_bs1024_epo300   -K 9 -run_name run1  -run_mode 0

# only run LDKNN (after fine-turning ImageNet-pretrained model with DefectMaker)  Resnet18
$ python main.py  -exp_name res18_dm_com6_bs1024_epo300    -K 9 -run_name run1  -run_mode 1


# DefectMaker + LDKNN  (REB)  Resnet18
$ python main.py  -exp_name res18_dm_com6_bs1024_epo300  -K 9  -run_name run1  -run_mode 2
```
</details>




<details open>
<summary>Training on Mvtec LOCO </summary>

Run commands below to reproduce results
on [Mvtec AD LOCO](https://www.mvtec.com/company/research/datasets/mvtec-loco)

1.  we don't use saliency for Mvtec LOCO 
    
2. modify the dataset path and OUTPUT path  in [global_param.py](src/global_param.py) according to your personal config.  
    the best K for Mvtec LOCO is 45. set K value in argparse or  [global_param.py](src/global_param.py)
    
3. Run commands below to train
```bash
$ cd  REB/projects/reb_mvtec_loco
#  Self-supervied learning  (fine-turning ImageNet-pretrained model with DefectMaker))  Resnet18
python main.py  -exp_name res18_dm_com6_bs1024_epo300  -K 45 -run_name run1  -run_mode 0

# only run LDKNN (after fine-turning ImageNet-pretrained model with DefectMaker)  Resnet18
$ python main.py  -exp_name res18_dm_com6_bs1024_epo300 -K 45  -run_name run1  -run_mode 1

# only run LDKNN ( directly use ImageNet-pretrained model)   Resnet18
$ python main.py  -exp_name res18_imagenet -K 45 -run_name run1  -run_mode 1

# DefectMaker + LDKNN  (REB)   Resnet18
$ python main.py  -exp_name res18_dm_com6_bs1024_epo300  -K 45 -run_name run1  -run_mode 2


# only run LDKNN ( directly use ImageNet-pretrained model)  Resnet18  
$ python main.py  -exp_name res18_imagenet   -K 45 -run_name run1  -run_mode 1

# only run LDKNN ( directly use ImageNet-pretrained model)  wideRes50 
$ python main.py  -exp_name wr50_imagenet   -K 45 -run_name run1  -run_mode 1

# only run LDKNN ( directly use ImageNet-pretrained model)  wideRes101 
$ python main.py  -exp_name wr101_imagenet   -K 45  -run_name run1  -run_mode 1

```
</details>







# Tutorials

##  Bezier_gen

* [visualize Bezier Curve](tutorials/bezier_curve.ipynb)
![](pictures/Bezier_gen.png)
##  DefectMaker
* [wrap multiple DefectMaker combinations with your data ](tutorials/DefectMakerUnion.ipynb)
![](pictures/dm.png)
