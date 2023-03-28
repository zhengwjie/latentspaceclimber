
# 首先，在潜空间中采样
# 求解特征向量，对其进行降维可视化
# 然后，开始使用pca对数据做降维



# 验证一下是不是概率密度大的地方，特征值会越大
# 画散点图就可以了

from operator import mod
import os
from signal import pthread_kill
import sys
import math
from this import d
from tkinter.messagebox import NO
from traceback import print_tb
import numpy as np
import time
from scipy.fftpack import shift
o_path = os.getcwd()

sys.path.append(o_path)

import torch
from matplotlib import image, pyplot as plt
from torchvision.utils import make_grid
import torch.autograd.functional as F
from functorch import make_functional,make_functional_with_buffers

# from torch_tools.visualization import to_image
from torch_tools.visualization import to_image
from visualization import inspect_all_directions
from loading import load_from_dir
import torch
from torchvision.transforms import Resize,Normalize
from torchvision.utils import make_grid
from PIL import Image
import io
import os

from torch_tools.visualization import to_image
from DirectionDiscovery.UMAP_neural import UMAP_Dense_Network

from utils1 import make_noise, one_hot
import lenet
import rob_predictor
import joblib
from scipy import stats
# define some constants
from DirectionDiscovery.vae_utils import load_G_D
from DirectionDiscovery import draw_graph

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# x 概率密度
x=[]
# y 前10的特征值求和
y=[]
model_name='SN_MNIST'
# model_name='stylegan2-ada_cifar10'
# global_generator,_=draw_graph.load_generator_descriminator(
#     model_name
# )

global_generator,_=draw_graph.load_generator('BigGAN_ImageNet')
number=2000
global_z=make_noise(number,global_generator.dim_z).to(device=device)
s1=[]
for i in range(number):
    z=global_z[i].unsqueeze(0)
    s,vh=draw_graph.get_feature_value_vector(global_generator,z)
    print(vh[0].shape)
    for j in range(3):
        s1.append(vh[j].reshape(1,-1))
s2=np.concatenate(s1)
# 正交分解
# print(s2.shape)

# from sklearn.manifold import TSNE

# ts = TSNE(n_components=2, init='pca', random_state=0)
from sklearn.decomposition import PCA
pca=PCA()

U,S,Vt = pca._fit_full(s2,2)

print(U.shape)
print(S)
print(Vt.shape)
z=make_noise(1,global_generator.dim_z).to(device=device)
images=[]
for i in range(15):
    for j in range(-5,5):
        shift1=j*torch.tensor(Vt[i]).to(device=device)
        image1=global_generator(z+shift1).squeeze(0)
        images.append(image1)
from torchvision.utils import make_grid,save_image

save_image(make_grid(images,nrow=10),"svd_pca_image4.png")

        

# print(result.shape)

# import matplotlib.pyplot as plt
# plt.scatter(result[:,0],result[:,1])
# plt.savefig("svd_pca2.png")

