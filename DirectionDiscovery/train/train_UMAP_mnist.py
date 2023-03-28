
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
import  DirectionDiscovery.UMAP_neural as umap_neural

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = umap_neural.UMAP_Dense_Network(in_dim=2,middle_dim=40,out_dim=5).to(device)
print(model)
# 模型准备好了

# dataset=torch.load("./DirectionDiscovery/train/trained_model/dataset_2_5.pt")
dataset=torch.load("./DirectionDiscovery/train/trained_model/dataset_50000.pt")
# dataset=torch.load("./DirectionDiscovery/train/trained_model/dataset_2_5_new.pt")
# dataset=torch.load("./DirectionDiscovery/train/trained_model/dataset_bigger.pt")
# dataset=torch.load("./DirectionDiscovery/train/trained_model/dataset.pt")
x,y=dataset[:][0],dataset[:][1]
print(torch.min(x))
print(torch.max(x))
x=((x-torch.min(x))/(torch.max(x)-torch.min(x))-0.5)*16.0
dataset=TensorDataset(x,y)
# torch.save(dataset,"./DirectionDiscovery/train/trained_model/dataset_2_5normalized.pt")
# torch.save(dataset,"./DirectionDiscovery/train/trained_model/dataset_normalized.pt")
torch.save(dataset,"./DirectionDiscovery/train/trained_model/dataset_50000_normalized.pt")
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset,test_dataset=random_split(dataset=dataset,lengths=[train_size,test_size])
dataloader=DataLoader(dataset=train_dataset,batch_size=4096,shuffle=True,drop_last=False)

learning_rate=0.1
momentum = 0.5
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn=MSELoss()
train_losses=[]
train_counter=[]
model=model.to(device)
# 加载数据，开始训练
# 应该要对输入数据做归一化
# 是不是也要对输出数据归一化
for epoch in range(300):
    for batch_idx,(in_data,out_data) in enumerate(dataloader):
        in_data=in_data.to(device)
        # print(torch.min(in_data))
        # print(torch.max(in_data))
        out_data=out_data.to(device)
        optimizer.zero_grad()
        output = model(in_data)
        loss = loss_fn(out_data.float(),output)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(in_data), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 512) + ((epoch - 1) * len(dataloader.dataset)))
    torch.save(model.state_dict(), './DirectionDiscovery/train/trained_model/UMAP_mnist_model_50000.pth')

# model.load_state_dict(torch.load('./DirectionDiscovery/train/trained_model/UMAP_mnist_model_5.pth'))
model.eval()

test_dataloader=DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,drop_last=False)

sum_loss=0
number=0
for batch_idx,(in_data,out_data) in enumerate(test_dataloader):
    in_data=in_data.to(device)
    out_data=out_data.to(device)
    # print(out_data)
    output=model(in_data)
    loss = loss_fn(out_data.float(),output)
    sum_loss+=loss.item()
    print(loss.item())
    number=number+1
    # print(output)
print(sum_loss/number)



# 接下来就是设计数据
# 使用UMAP获取数据
# 开始训练：设计loss function,使用梯度下降算法
# 最后保存训练好的模型