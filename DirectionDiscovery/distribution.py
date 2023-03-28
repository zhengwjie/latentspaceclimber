
from cmath import tanh
from torch import nn
import torch

# 2维到高维正态分布
class Uniform_to_latent_Network(nn.Module):
    def __init__(self,in_dim=2,middle_dim=512,out_dim=128) -> None:
        super().__init__()
        self.linear_layers=nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features=in_dim,out_features=middle_dim),
            nn.BatchNorm1d(middle_dim),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
            
            # nn.Dropout(0.5),
            nn.Linear(middle_dim,middle_dim),
            nn.BatchNorm1d(middle_dim),
            nn.LeakyReLU(0.1),
            # nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(middle_dim,middle_dim),
            nn.BatchNorm1d(middle_dim),
            nn.LeakyReLU(0.05),
            # nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(middle_dim,middle_dim),
            nn.BatchNorm1d(middle_dim),
            nn.LeakyReLU(0.05),
            # nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(middle_dim,out_features=out_dim),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.linear_layers(x)
        x=x
        return x

if __name__=="__main__":
    from operator import mod
    import os,sys
    from textwrap import indent
    o_path = os.getcwd()
    import torch
    sys.path.append(o_path)
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from torch.utils.data import TensorDataset,DataLoader
    from torch.nn import MSELoss,CrossEntropyLoss
    # 首先构建3维的点
    from umap import UMAP
    from torch import optim
    from torch.utils.data import random_split


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")


    model = Uniform_to_latent_Network().to(device)
    print(model)
    # 模型准备好了
    # 无需准备数据

    learning_rate=0.1
    momentum = 0.5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # loss_fn=nn.KLDivLoss(reduction="batchmean")
    loss_fn=nn.MSELoss(reduction="batchmean")
    train_losses=[]
    train_counter=[]
    model=model.to(device)
    # 加载数据，开始训练
    # 应该要对输入数据做归一化
    # 是不是也要对输出数据归一化
    out_data=torch.randn(20000,128).to(device=device)
    in_data=((torch.rand((20000,2))*160)-80).to(device=device)
    for epoch in range(3000):
            # print(torch.min(in_data))
            # print(torch.max(in_data))
        in_data=((torch.rand((20000,2))*160)-80).to(device=device)
        optimizer.zero_grad()
        output = model(in_data)
        # print(torch.max(output,dim=-1))
        mean=((torch.mean(output))**2).sum()
        std=((torch.std(output)-1)**2).sum()
        
        loss = mean+std
        loss.backward()
        optimizer.step()
        print(loss.item())
    torch.save(model.state_dict(), './DirectionDiscovery/train/trained_model/1.pth')


