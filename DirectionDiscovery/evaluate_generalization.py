# 验证jacobian矩阵是不是处处近似
import os
import sys
o_path = os.getcwd()
os.chdir("/home/zhengwanjie/GANLatentDiscovery-master")
o_path = os.getcwd()
sys.path.append(o_path)
print(o_path)
import torch
from matplotlib import pyplot as plt
from utils1 import make_noise
torch.cuda.empty_cache() # PyTorch thing
print('122')
# from visualization import interpolate
from loading import load_from_dir
print('122')
from DirectionDiscovery.draw_graph import jacobian_G

print('122')
# SN_MNIST  img.shape=[1*32*32]  G.dim_z=128 directions=128
# SN_Anime  img.shape=[3*64*64] G.dim_z=128 directions=128
# BIGGAN ILSVRC img.shape=[3*128*128] G.dim_z=120 direction=120
# ProgGAN CelebA-HQ  img.shape=[3*1024*1024] G.dim_z=[1*1*512] directions=512
# StyleGAN img.shape=[3*1024*1024] G.dim_z=512 
# deformator.input_dim !=方向数量

device = "cuda" if torch.cuda.is_available() else "cpu"
print(123)

# deformator, G, shift_predictor = load_from_dir(
#     'models/pretrained/deformators/stylegan2-ada/',
#     G_weights='models/pretrained/generators/stylegan2-ada/cifar10u-cifar-ada-best-fid.pkl',bias=False)
deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/SN_MNIST/',
    G_weights='./models/pretrained/generators/SN_MNIST/')
# deformator, G, shift_predictor = load_from_dir("./models/pretrained/deformators/BigGAN/",
# G_weights="./models/pretrained/generators/BigGAN/G_ema.pth")
print("加载完成")
G=G.to(device)
z=make_noise(1, G.dim_z).to(device=device)
jacobian=jacobian_G(G,z)


final_result=[]
for i in range(100):
    z_i=make_noise(1, G.dim_z).to(device=device)
    jacobian_i=jacobian_G(G,z_i)
    val_i=torch.norm(jacobian-jacobian_i,p=2).item()
    final_result.append(val_i)
    print(val_i)
plt.hist(final_result)






