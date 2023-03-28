

import os
import sys
o_path = os.getcwd()
os.chdir("/home/zhengwanjie/GANLatentDiscovery-master")
o_path = os.getcwd()
sys.path.append(o_path)
print(o_path)
from tkinter.messagebox import NO
from matplotlib.cbook import flatten
from numpy import imag
import torch
from matplotlib import pyplot as plt
from zmq import device
from utils1 import make_noise

from torch_tools.visualization import to_image
# from visualization import interpolate
from loading import load_from_dir
from utils1 import make_noise, one_hot
import torch.autograd.functional as F
from functools import partial
from functorch import make_functional,make_functional_with_buffers

# SN_MNIST  img.shape=[1*32*32]  G.dim_z=128 directions=128
# SN_Anime  img.shape=[3*64*64] G.dim_z=128 directions=128
# BIGGAN ILSVRC img.shape=[3*128*128] G.dim_z=120 direction=120
# ProgGAN CelebA-HQ  img.shape=[3*1024*1024] G.dim_z=[1*1*512] directions=512
# StyleGAN img.shape=[3*1024*1024] G.dim_z=512 
# deformator.input_dim !=方向数量

device = "cuda" if torch.cuda.is_available() else "cpu"
deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/SN_MNIST/',
    G_weights='./models/pretrained/generators/SN_MNIST/')
# deformator=deformator.cuda()

# 查看神经网络的参数
# deformator不用backward求梯度，只需要用参数就可以  
print(tuple(deformator.parameters()))
# # tuple(G.parameters())
# input_tensor=torch.zeros(deformator.input_dim,requires_grad=True).cuda()
#     8: fat
# 2: roundness
# 10: xscale
# 13: zoom
# 44: rotation
indices=torch.tensor([2,8,10,13,44])
jacobian_deformator=tuple(deformator.parameters())[0].T
jacobian_selected=torch.index_select(jacobian_deformator,1,indices).cuda()
print(jacobian_selected.shape)

# 5*128
# 接下来计算G产生的jacobian矩阵
# 把它们乘起来在一起
# torch.autograd
G=G.cuda()
G.eval()

z=make_noise(1, G.dim_z).cuda()
func, params, buffers= make_functional_with_buffers(G)

# print(func)
jacobian_G=F.jacobian(partial(func,params,buffers),z)
# 所以目前求出来的矩阵结果是对的
# 初步估计
# print(jacobian_G.sum())
jacobian_G=jacobian_G.squeeze()
print(jacobian_G.shape)

jacobian_G=jacobian_G.reshape(-1,jacobian_G.shape[-1])
print(jacobian_G.shape)
# 使用SVD去发现模式

full_jacobian=torch.mm(jacobian_G,jacobian_selected)

print(full_jacobian.shape)

import numpy as np

u, s, vh = np.linalg.svd(full_jacobian.cpu().numpy(), full_matrices=True)





