

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
from torch.autograd.functional import jacobian
# SN_MNIST  img.shape=[1*32*32]  G.dim_z=128 directions=128
# SN_Anime  img.shape=[3*64*64] G.dim_z=128 directions=128
# BIGGAN ILSVRC img.shape=[3*128*128] G.dim_z=120 direction=120
# ProgGAN CelebA-HQ  img.shape=[3*1024*1024] G.dim_z=[1*1*512] directions=512
# StyleGAN img.shape=[3*1024*1024] G.dim_z=512 
# deformator.input_dim !=方向数量

def test1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deformator, G, shift_predictor = load_from_dir(
        './models/pretrained/deformators/SN_MNIST/',
        G_weights='./models/pretrained/generators/SN_MNIST/')
    print(G)
    print(deformator)
    G=G.to(device)
    z = make_noise(1, G.dim_z).to(device=device)
    print(z.shape)
    image=G(z)
    # image.backward()
    print(image)
    deformator.eval()
    input_tensor=torch.zeros(deformator.input_dim,requires_grad=True).to(device=device)
    deformator=deformator.to(device)
    input_tensor[3]=1.0
    output=deformator(input_tensor)

    input_tensor.retain_grad()
    gradients=torch.ones(output.shape,requires_grad=True).to(device=device)
    output.backward(gradient=gradients)
    print(input_tensor.grad)

def forward_function(deformator,G,input_tensor,z):
    deformator=deformator.to(device)
    G=G.to(device)
    input_tensor=input_tensor.to(device)
    z=z.to(device)
    input_tensor.reshape(1,input_tensor.shape[0])
    latent_shift=deformator(input_tensor)
    shifted_image = G.gen_shifted(z, latent_shift)
    shifted_image=flatten(shifted_image)
    return shifted_image


def mul(x):
    return 3.0*x[0]+3.4*x[1]
from DirectionDiscovery import draw_graph
if __name__=='__main__':
    import pickle
    with open("./final_z0.pkl","rb") as f:

        final_z0=pickle.load(f)
    print(final_z0[24:25])
    print(final_z0[24:25].shape)
    with open("./z.pkl","rb") as f:
        z=pickle.load(f)
    print(z)
    print(z.shape)
    global_generator, _ = draw_graph.load_generator_descriminator(
        "cifar10"
    )
    img0=global_generator(final_z0[24:25])
    print(img0)
    img1=global_generator(final_z0[24:25])
    print(img1)
    if torch.allclose(final_z0[24:25],z):
        print("hhh")
    if torch.allclose(img0,img1):
        print("hhh")
    new_img=img1-img0
    print(new_img)

    pass

    

















# deformator.input_dim=128 表示有128个方向

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/SN_Anime/',
#     G_weights='./models/pretrained/generators/SN_Anime/')
# print(deformator.input_dim)
# print(G.dim_z)
# z = make_noise(1, G.dim_z, None).cuda()
# img=G(z)
# print(img.shape)
# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/BigGAN/',
#     G_weights='./models/pretrained/generators/BigGAN/G_ema.pth')

# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/ProgGAN/',
#     G_weights='./models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth')
# print(deformator.input_dim)
# print(G.dim_z)
# z = make_noise(1, G.dim_z, None).cuda()
# img=G(z)
# print(img.shape)
# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/StyleGAN2/',
#     G_weights='./models/pretrained/generators/StyleGAN2/stylegan2-ffhq-config-f.pt')
# print(deformator.input_dim)
# print(G.dim_z)
# z = make_noise(1, G.dim_z, None).cuda()
# img=G(z)
# print(img.shape)
# discovered_annotation = ''
# for d in deformator.annotation.items():
#     discovered_annotation += '{}: {}\n'.format(d[0], d[1])
# print('human-annotated directions:\n' + discovered_annotation)

# import pickle
# import stylegan2_ada
# import sys
# sys.path.append("../stylegan2_ada/torch_utils")
# sys.path.append("~/GANLatentDiscovery-master/models/stylegan2_ada/")
# print(sys.path)
# G_weights="models/pretrained/generators/stylegan2-ada/cifar10.pkl"
# print("start loading")
# G=stylegan2_ada.load_interface.load_pkl_gan(G_weights=G_weights)
# print("end")
# import torch
# import sys
# from models import gan_load
# G_weights="models/pretrained/generators/stylegan2-ada/cifar10u-cifar-ada-best-fid.pkl"
# G=gan_load.make_style_gan2_ada(G_weights=G_weights)
# z = torch.randn([1, G.z_dim]).cuda()
# print(G.c_dim)
# c=None
# print(sys.path)
# img=G(z,c)
# print(sys.path)
# print(img.shape)
# print(sys.path)
# deformator, G, shift_predictor = load_from_dir(
#     './models/pretrained/deformators/stylegan2-ada/',
#     G_weights='./models/pretrained/generators/stylegan2-ada/cifar10.pkl')
# print(deformator.input_dim)