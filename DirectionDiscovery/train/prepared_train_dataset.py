from operator import mod
import os,sys
o_path = os.getcwd()
import torch
sys.path.append(o_path)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from torch.utils.data import TensorDataset,DataLoader
from torch.nn import MSELoss
# 首先构建3维的点
from umap import UMAP
from torch import optim

import  DirectionDiscovery.UMAP_neural as umap_neural

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model = umap_neural.UMAP_Dense_Network(in_dim=2,middle_dim=8,out_dim=5).to(device)
# print(model)
# 模型准备好了
# 准备5维的数据
data=[[0,0,0,0,0]]
max_value=8
internal=1
for i in range(-max_value,max_value+1,internal):
    for j in range(-max_value,max_value+1,internal):
        for k in range(-max_value,max_value+1,internal):
            for l in range(-max_value,max_value+1,internal):
                for m in range(-max_value,max_value+1,internal):
                    # new_data=[[i/10,j/10,k/10,l/10,m/10]]
                    new_data=[[i,j,k,l,m]]
                    data=np.vstack((data,new_data))
print(data.shape)
# 已经准备好数据了
# 开始使用UMAP对数据进行降维
reducer = UMAP(random_state=32,min_dist=internal,spread=internal)
result = reducer.fit_transform(data)
print(result.shape)
# 将数据转化成tensor向量并转化为数据集

x,y=torch.from_numpy(result),torch.from_numpy(data)
dataset=TensorDataset(x,y)
torch.save(dataset,"./DirectionDiscovery/train/trained_model/dataset_2_5.pt")