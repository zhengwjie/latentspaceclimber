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

from utils1 import make_noise, one_hot



# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/SN_Anime/',
#     G_weights='./models/pretrained/generators/SN_Anime/')
with torch.no_grad():
    # deformator, G, shift_predictor = load_from_dir(
    # "./models/pretrained/deformators/BigGAN/",
    # G_weights="./models/pretrained/generators/BigGAN/G_ema.pth")
    
    deformator, G, shift_predictor = load_from_dir(
    './models/pretrained/deformators/SN_MNIST/',
    G_weights='./models/pretrained/generators/SN_MNIST/')
    deformator.eval()





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
    directions = [8, 13]
    angles = [0,90]
    # z是原图的特征
    z = make_noise(1, G.dim_z).cuda()
    shifts_r = 8.0
    # shift_count默认为5
    shifts_count = 5
    shifted_images = []
    for x in range(-shifts_count, shifts_count+1, 1):
        row_imgs=[]
        for y in range(shifts_count, -shifts_count-1, -1):
            if(math.sqrt(x * x + y * y)==0):
                original_img = G(z).cpu()[0]
                print(original_img.shape)
                row_imgs.append(original_img)
                continue
            deformator_input=one_hot(deformator.input_dim, x/shifts_count*shifts_r, directions[0]).cuda() + one_hot(deformator.input_dim, y/shifts_count*shifts_r, directions[1]).cuda()
            if deformator is not None:
                latent_shift = deformator(deformator_input)
            else:
                latent_shift = deformator_input
            shifted_image = G.gen_shifted(z, latent_shift).cpu()[0]
            print(shifted_image.shape)
            row_imgs.append(shifted_image)
        shifted_images.append(row_imgs)
    imgs=[]
    rows_count=2*shifts_count+1
    fig, axs = plt.subplots(rows_count,figsize=(20, 20))

    for ax, row_imgs in zip(axs, shifted_images):
        ax.axis('off')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.1, hspace=0.1)
        ax.imshow(to_image(make_grid(row_imgs, nrow=rows_count, padding=1), True))
    fig.canvas.draw()
    plt.close(fig)
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # crop borders
    nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
    img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
    imgs.append(img)
    out_file = "./test.jpg"
    print('saving chart to {}'.format(out_file))
    Image.fromarray(np.hstack(imgs)).save(out_file)
 