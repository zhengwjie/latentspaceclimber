# from cmath import tanh
# from turtle import forward
from cmath import tanh
from torch import nn
# 构建一个全连接的神经网络

# 2维到高维
class UMAP_Dense_Network(nn.Module):
    def __init__(self,in_dim,middle_dim,out_dim) -> None:
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
        x=x*8.0
        return x


