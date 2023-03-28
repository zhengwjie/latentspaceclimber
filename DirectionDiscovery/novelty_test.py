

# 这里是用来计算novelty的工具

import torchvision.models as models
# 加载  wide resnet 50
from torchvision import datasets, transforms
import torch
from sklearn.neighbors import NearestNeighbors
import joblib
import numpy as np
import os,sys


o_path = "/home/zhengwanjie/GANLatentDiscovery-master/"

sys.path.append(o_path)
os.chdir(o_path)
from torch_tools.visualization import to_image
# 明天的任务
from DirectionDiscovery.vae_utils import load_G_D
from utils1 import make_noise
# 找模型 使用resnet18抽特征
# 测距离 
# 算novelty score
# 把novelty score 集成到系统中
# clap(0,1)
from collections import OrderedDict
from DirectionDiscovery import novelty

device = "cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def test_novelty_map_pixel_level():
    generator,_=load_G_D("cifar10")
    generator.to(device)

    z=make_noise(1,generator.dim_z).to(device=device)
    images=generator(z)

    model_name="cifar10"
    feature_extractor=novelty.load_model()
    n_neighbors=15
    nbrs=novelty.load_knn_model("cifar10")
    scores,score_map_list,image_neighbors=novelty.novelty_map_pixel_level(images,model_name,feature_extractor,n_neighbors,nbrs)
    print(scores)
    print(score_map_list[0].shape)
    print(image_neighbors)
    

if __name__=="__main__":
    test_novelty_map_pixel_level()
    # dataset=novelty.load_knn_dataset("cifar10")

