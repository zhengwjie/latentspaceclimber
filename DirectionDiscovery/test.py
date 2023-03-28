# import os
# from signal import pthread_kill
# import sys
# import math
# import numpy as np

# o_path = os.getcwd()

# sys.path.append(o_path)

# import torch
# from matplotlib import pyplot as plt
# from torchvision.utils import make_grid

# # from torch_tools.visualization import to_image
# from torch_tools.visualization import to_image
# from visualization import interpolate
# from loading import load_from_dir
# import torch
# from torchvision.transforms import Resize
# from torchvision.utils import make_grid
# from PIL import Image
# import io
# import os

# from torch_tools.visualization import to_image

# from utils import is_conditional, make_noise, one_hot


# # deformator, G, shift_predictor = load_from_dir(
# #     './models/pretrained/deformators/SN_MNIST/',
# #     G_weights='./models/pretrained/generators/SN_MNIST/')

# # deformator, G, shift_predictor = load_from_dir(
# #     './models/pretrained/deformators/SN_Anime/',
# #     G_weights='./models/pretrained/generators/SN_Anime/')
# with torch.no_grad():
#     deformator, G, shift_predictor = load_from_dir(
#     "./models/pretrained/deformators/BigGAN/",
#     G_weights="./models/pretrained/generators/BigGAN/G_ema.pth",)
#     deformator.eval()
    


# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/ProgGAN/',
#     G_weights='./models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth')

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/StyleGAN2/',
#     G_weights='./models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt')

# discovered_annotation = ''
# for d in deformator.annotation.items():
#     discovered_annotation += '{}: {}\n'.format(d[0], d[1])
# print('human-annotated directions:\n' + discovered_annotation)
# 可以跑通代码了
# 接下来就是构建图片
# print(deformator.type)
# 这里的 deformator.type=DeformatorType.ORTHO

    # directions = [22, 24, 59, 60, 75, 81]
    # angles = [15, 45, 75, 105, 135, 165]
    # # z是原图的特征
    # z = make_noise(1, G.dim_z).cuda()
    # shifts_r = 8.0
    # # shift_count默认为5
    # shifts_count = 3
    # shifted_images = []
    # for x in range(-shifts_count, shifts_count+1, 1):
    #     row_imgs=[]
    #     for y in range(shifts_count, -shifts_count-1, -1):
    #         if(math.sqrt(x * x + y * y)==0):
    #             original_img = G(z).cpu()[0]
    #             print(original_img.shape)
    #             row_imgs.append(original_img)
    #             continue
    #         angle = math.acos(x / math.sqrt(x * x + y * y)) * 180 / math.pi
    #         if(y<0):
    #             angle=180-angle
    #         # 这个是要求权重
    #         start_angle = ((angle - 15) // 30) * 30 + 15
    #         end_angle = start_angle + 30
    #         print(angle)
    #         start_direction = directions[int((angle - 15) // 30)%6]
    #         end_direction = directions[int((angle - 15) // 30 + 1)%6]
    #         shift=8*math.sqrt((x*x+y*y)/18)
    #         start_shift=shift
    #         end_shift=shift
    #         if(angle<15):
    #             start_shift=-start_shift
    #         if(angle>165):
    #             end_shift=-end_shift
    #         start_weight=(angle-start_angle)/30
    #         end_weight=1-start_weight
    #         if(y<0):
    #             start_shift=-start_shift
    #             end_shift=-end_shift
    #         # 求 0-180 之间的角度
    #         # 求 w1 和 w2 (w2=1-w1)
    #         # 如果deformator非空，就需要deformator转
    #         # shift这个值应该是半径
    #         # dim是方向维度
    #         deformator_input=start_weight*one_hot(deformator.input_dim, start_shift, start_direction).cuda() + end_weight*one_hot(deformator.input_dim, end_shift, end_direction).cuda()
    #         if deformator is not None:
    #             latent_shift = deformator(deformator_input)
    #         else:
    #             latent_shift = deformator_input
    #         shifted_image = G.gen_shifted(z, latent_shift).cpu()[0]
    #         print(shifted_image.shape)
    #         row_imgs.append(shifted_image)
    #     shifted_images.append(row_imgs)
    # imgs=[]
    # rows_count=2*shifts_count+1
    # fig, axs = plt.subplots(rows_count,figsize=(20, 20))

    # for ax, row_imgs in zip(axs, shifted_images):
    #     ax.axis('off')
    #     plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
    #             wspace=0.1, hspace=0.1)
    #     ax.imshow(to_image(make_grid(row_imgs, nrow=rows_count, padding=1), True))
    # fig.canvas.draw()
    # plt.close(fig)
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # # crop borders
    # nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
    # img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
    # imgs.append(img)
    # out_file = "./test4.jpg"
    # print('saving chart to {}'.format(out_file))
    # Image.fromarray(np.hstack(imgs)).save(out_file)
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
from DirectionDiscovery.draw_graph import draw_svd_feature_vector,show


# SN_MNIST  img.shape=[1*32*32]  G.dim_z=128 directions=128
# SN_Anime  img.shape=[3*64*64] G.dim_z=128 directions=128
# BIGGAN ILSVRC img.shape=[3*128*128] G.dim_z=120 direction=120
# ProgGAN CelebA-HQ  img.shape=[3*1024*1024] G.dim_z=[1*1*512] directions=512
# StyleGAN img.shape=[3*1024*1024] G.dim_z=512 
# deformator.input_dim !=方向数量

device = "cuda" if torch.cuda.is_available() else "cpu"
# deformator, G, shift_predictor = load_from_dir(
#     'models/pretrained/deformators/stylegan2-ada/',
#     G_weights='models/pretrained/generators/stylegan2-ada/cifar10u-cifar-ada-best-fid.pkl',bias=False)
deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/SN_MNIST/',
    G_weights='./models/pretrained/generators/SN_MNIST/')
# deformator, G, shift_predictor = load_from_dir("./models/pretrained/deformators/BigGAN/",
# G_weights="./models/pretrained/generators/BigGAN/G_ema.pth")

G=G.to(device)
z=make_noise(1, G.dim_z).to(device=device)
imgs=draw_svd_feature_vector(G,z,-5,5,9)
show(imgs,"result5.png")

# import torch
# # 加载模型
# import os
# import sys
# o_path = os.getcwd()
# os.chdir("/home/zhengwanjie/GANLatentDiscovery-master")
# o_path = os.getcwd()
# sys.path.append(o_path)
# print(o_path)
# from tkinter.messagebox import NO
# from matplotlib.cbook import flatten
# from numpy import imag
# import torch
# from matplotlib import pyplot as plt
# from zmq import device
# from utils import make_noise

# from torch_tools.visualization import to_image
# # from visualization import interpolate
# from loading import load_from_dir
# from utils import make_noise, one_hot

# # SN_MNIST  img.shape=[1*32*32]  G.dim_z=128 directions=128
# # SN_Anime  img.shape=[3*64*64] G.dim_z=128 directions=128
# # BIGGAN ILSVRC img.shape=[3*128*128] G.dim_z=120 direction=120
# # ProgGAN CelebA-HQ  img.shape=[3*1024*1024] G.dim_z=[1*1*512] directions=512
# # StyleGAN img.shape=[3*1024*1024] G.dim_z=512 
# # deformator.input_dim !=方向数量

# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# deformator, G, shift_predictor = load_from_dir("./models/pretrained/deformators/BigGAN/",
# G_weights="./models/pretrained/generators/BigGAN/G_ema.pth")
# G.to(device)
# z = make_noise(1, G.dim_z).to(device=device)
# img=G(z)
# print(img.shape)
# # 先把有意义的那些方向向量找出来
# # 再把雅可比矩阵分解之后的特征向量找出来
# # 使用线性回归，看看效果
# from DirectionDiscovery.draw_graph import get_feature_value_vector
# # classes = G.mixed_classes(z.shape[0]).to(device)
# import torch.autograd.functional as F

# from functools import partial
# from functorch import make_functional,make_functional_with_buffers
# func, params, buffers= make_functional_with_buffers(G.big_gan)
# print(G.big_gan.shared(G.mixed_classes(1).to(device)))

# jacobian_GAN=F.jacobian(partial(func,params,buffers),(z,G.big_gan.shared(G.mixed_classes(1).to(device))))
# print(jacobian_GAN)