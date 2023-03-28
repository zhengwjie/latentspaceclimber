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
import joblib




# model = umap_neural.UMAP_Dense_Network(in_dim=2,middle_dim=8,out_dim=5).to(device)
# print(model)
# 模型准备好了
# 准备5维的数据
data=np.random.uniform(-8,8,(50000,5))
# 已经准备好数据了
# 开始使用UMAP对数据进行降维
reducer = UMAP(random_state=32)
result = reducer.fit_transform(data)

#reducer是一个UMAP降维模型
joblib.dump(reducer, './DirectionDiscovery/train/trained_model/umap.model')
# 加载模型的办法
# reducer = joblib.load('./DirectionDiscovery/train/trained_model/umap.model')


print(result.shape)
# 将数据转化成tensor向量并转化为数据集

x,y=torch.from_numpy(result),torch.from_numpy(data)
dataset=TensorDataset(x,y)
torch.save(dataset,"./DirectionDiscovery/train/trained_model/dataset_50000.pt")