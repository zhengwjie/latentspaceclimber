import os
from signal import pthread_kill
import sys
import math
import numpy as np

o_path = os.getcwd()

sys.path.append(o_path)

import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

# from torch_tools.visualization import to_image
from torch_tools.visualization import to_image

from visualization import interpolate
from loading import load_from_dir
import torch
from torchvision.transforms import Resize
from torchvision.utils import make_grid
from PIL import Image
import io
import os

from torch_tools.visualization import to_image
from visualization import gen_animation
from utils1 import make_noise, one_hot


# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/SN_MNIST/',
#     G_weights='./models/pretrained/generators/SN_MNIST/')

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/SN_Anime/',
#     G_weights='./models/pretrained/generators/SN_Anime/')
with torch.no_grad():
    # deformator, G, shift_predictor = load_from_dir(
    # "./models/pretrained/deformators/BigGAN/",
    # G_weights="./models/pretrained/generators/BigGAN/G_ema.pth",)
    # deformator.eval()

    deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/SN_MNIST/',
    G_weights='./models/pretrained/generators/SN_MNIST/')
    deformator.cuda()
    G.cuda()
    # This is used to look at one direction
    # [-5,+5]，show its changes
    # 22: zoom
    # 24: lighting
    # 59: background removal
    # 60: background_blur
    # 75: rotation
    # 81: y_shift

    # sn_mnist

#     8: fat
# 2: roundness
# 10: xscale
# 13: zoom
# 44: rotation
    gen_animation(G,deformator,8,'./direction_8_n.gif',r=8)
    gen_animation(G,deformator,2,'./direction_2_n.gif',r=8)
    gen_animation(G,deformator,10,'./direction_10_n.gif',r=8)
    gen_animation(G,deformator,13,'./direction_13_n.gif',r=8)
    gen_animation(G,deformator,44,'./direction_44_n.gif',r=8)
    gen_animation(G,deformator,81,'./direction_81_n.gif',r=8)
    gen_animation(G,deformator,82,'./direction_82_n.gif',r=8)
    gen_animation(G,deformator,83,'./direction_83_n.gif',r=8)

    # gen_animation(G,deformator,8,'./direction_8.gif',r=8)
    # gen_animation(G,deformator.linear,2,'./direction_2.gif',r=8)
    # gen_animation(G,deformator.linear,10,'./direction_10.gif',r=8)
    # gen_animation(G,deformator.linear,13,'./direction_13.gif',r=8)
    # gen_animation(G,deformator.linear,44,'./direction_44.gif',r=8)

    for i in range(64):
        gen_animation(G,deformator,i,"./results/dir_"+str(i)+".gif",r=8)