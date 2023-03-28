from tkinter.tix import Tree
import torch
from VAE_train import VAE,save_image, test,vae_path
# build model
vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
# 加载保存的权重

vae.load_state_dict(
    torch.load(vae_path))
if torch.cuda.is_available():
    vae.cuda()
# test(vae=vae)
# 测试vae的fid为多少
# Frechet inception distance 
def fid():

    return 
def samples_to_image(samples,dataset_target_path,start_index=0):
    import os
    o_path = os.getcwd()
    import sys
    sys.path.append(o_path)
    
    from torch_tools.visualization import to_image
    for i in range(len(samples)):
        img=to_image(samples[i],True)
        img=img.resize((32,32))
        img.save(dataset_target_path+"/"+str(i+start_index)+".png")
    
with torch.no_grad():
    z = torch.randn(6000, 2).cuda()
    sample = vae.decoder(z).cuda()
    sample=sample.reshape(-1,1,28,28)
    samples_to_image(sample,"data/mnist_4")