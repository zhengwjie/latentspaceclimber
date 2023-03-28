


# 首先，使用partial_forward
from unicodedata import decimal
import torch
from vae_utils import load_G_D,samples_to_image
import sys
sys.path.append("/home/zhengwanjie/GANLatentDiscovery-master")
# # 加载SN_MNIST
from draw_graph import load_generator_descriminator

from utils1 import make_noise
device = "cuda" if torch.cuda.is_available() else "cpu"
def partial_forward(generator,layer_name):
    # 测试一下partial forward是不是正确的
    return
# model=['SN_MNIST']
def generate_samples(model_name,samples_path,sample_number=6000,target_size=28):
    G,_=load_generator_descriminator(model_name)
    G=G.to(device=device)
    for i in range(sample_number//100):
        z=make_noise(100,G.dim_z).to(device)
        with torch.no_grad():
            imgs=G(z)
        samples_to_image(imgs,samples_path,start_index=i*100,target_size=target_size)
    if sample_number%100==0:
        return
    z=make_noise(sample_number%100,G.dim_z).to(device)
    with torch.no_grad():
        imgs=G(z)
    samples_to_image(imgs,samples_path,start_index=(sample_number//100)*100,target_size=target_size)       
def generate_sample_using_2D(model_name,samples_path,sample_number=6000,scale=80):
    if model_name=='gaussian_mnist':
        G1,_=load_G_D('gaussian')
        G2,_=load_G_D('mnist')
    G1=G1.to(device=device)
    G2=G2.to(device=device)
    G1.eval()
    G2.eval()
    for i in range(sample_number//100):
        z=make_noise(100,G1.dim_z).to(device)
        z=z*scale
        with torch.no_grad():
            noise=G1(z)
            noise=noise.reshape(-1,G2.dim_z)
            noise=noise*6.0
            imgs=G2(noise)
        samples_to_image(imgs,samples_path,start_index=i*noise.size(0))
    if sample_number%100==0:
        return
    z=make_noise(sample_number%100,G1.dim_z).to(device)
    z=z*scale
    with torch.no_grad():
        noise=G1(z)
        noise=noise.reshape(-1,G2.dim_z)
        noise=noise*6.0
        imgs=G2(noise)
    samples_to_image(imgs,samples_path,start_index=(sample_number//100)*100)
def generate_2D_graphs(model_name,sample_path,scale=80):
    import torchvision.utils as vutils
    if model_name=='gaussian_mnist':
        G1,_=load_G_D('gaussian')
        G2,_=load_G_D('mnist')
    G1=G1.to(device=device)
    G2=G2.to(device=device)
    G1.eval()
    G2.eval()
    final_images=torch.randn(0,1,32,32).to(device=device)
    for i in range(-scale*3,scale*3,5):
        for j in range(-scale*3,scale*3,5):
            input=torch.Tensor([[i+0.1,j+0.1]]).to(device=device)
            with torch.no_grad():
                noise=G1(input)
                # print(noise.shape)
                noise=noise.reshape(1,-1)
                # print(noise.shape)
                noise=noise*6.0
                imgs=G2(noise)
                # print(imgs.shape)
                final_images=torch.cat((final_images,imgs),dim=0)
            # return
    print(final_images.shape)
    samples_to_image(final_images,sample_path)
    vutils.save_image(vutils.make_grid(final_images.data,nrow=36*5),'./data/test3.png')


def dataset_to_samples(dataset_name,samples_path,dataset_root='./mnist_data/',train=False,sample_number=6000,target_size=28):
    from torchvision import datasets, transforms
    from vae_utils import samples_to_image
    from torch.utils.data import TensorDataset
    if dataset_name=='MNIST':
        dataset = datasets.MNIST(root=dataset_root, train=train, transform=transforms.ToTensor(), download=True)
    elif dataset_name=='cifar10':
        dataset = datasets.CIFAR10(root=dataset_root, train=train, transform=transforms.ToTensor(), download=True)
        print(len(dataset))
            # Data Loader (Input Pipeline)
    elif dataset_name=='gaussian':
        # torch.manual_seed(20)
        # torch.cuda.manual_seed_all(20)
        samples=torch.randn((sample_number,1,32,32))
        samples=samples/torch.max(abs(samples))
        label=torch.ones(sample_number)
        dataset=TensorDataset(samples,label)
    
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)
    for batch_idx, (data, _) in enumerate(dataset_loader):
        print(batch_idx)
        samples_to_image(data,samples_path,start_index=batch_idx*100,target_size=target_size)

# 检测fid的命令如下
# python -m pytorch_fid data/mnist_1 data/mnist_2
# 现有的gan和
# 论文里面报告的cifar10的fid是
if __name__=='__main__':
    # generate_samples('SN_MNIST','./data/mnist_1')
    # generate_samples('mnist','./data/mnist_100',target_size=32,sample_number=1210,scale=1)
    
    # generate_samples('mnist','./data/mnist/mnist_sample',sample_number=4000,target_size=32)
    
    # dataset_to_samples('mnist','./data/mnist/mnist_test',dataset_root='./data/datasets/mnist/',train=False,target_size=32)
    dataset_to_samples('cifar10','./data/cifar10/cifar10_test',dataset_root='./data/datasets/cifar10/',train=False)
    # generate_samples('stylegan2-ada_cifar10',
    # './data/cifar10/cifar10_generator_100000',
    # sample_number=100000)
    # dataset_to_samples('gaussian','./data/gaussian/gaussian_test_2')
    # generate_samples('mnist','./data/mnist_samples',scale=80.0)
    # generate_samples('gaussian','./data/gaussian/gaussian_generate_sigma_200',sample_number=6000,scale=200)
    # # 'gaussian_mnist'
    # # 
    # generate_2D_graphs('gaussian_mnist',scale=30,sample_path='./data/gaussian_mnist/generated_samples_1')
    # generate_sample_using_2D('gaussian_mnist','./data/gaussian_mnist/generated_samples',scale=30,sample_number=6000)


# fid是在与测试集进行对比的结果
# 直接生成图片
# 2维潜空间生成的图片的fid是 15.2555381652769 400个epoch,sigma=80
# 把sigma增加到200，看看fid是否会提升

# 生成高斯分布的图片
# sigma=80 epoch=400 fid=33.46294119968881
# sigma=200 epoch=350 fid=25.413655948701447
# sigma=200 epoch=400 fid=20.37352034762972  (图片从6000增加到10000 fid基本不变化)
# sigma=200 epoch=450 fid=27.21171805018905
# sigma=200 epoch=500 fid=23.341005060937007

# sigma=200 epoch=650 fid=15.529575049409317
# sigma=200 epoch=700 fid=15.348314471310559
# sigma=200 epoch=750 fid=19.247637690290297
# sigma=200 epoch=800 fid=20.002142771519047
# sigma=200 epoch=850 fid=20.22749829116067
# sigma=200 epoch=900 fid=36.73012181984375
# sigma=200 epoch=950 fid=39.20132727668767
# sigma=200 epoch=1000 fid=35.42558812176783
# sigma=200 epoch=1000 fid=35.42558812176783


# 2维空间--->高斯噪声---->图片  fid=32.18540223241379
# 2维空间--->高斯噪声---->图片  fid=15.612329175861873


# mnist 随机噪声实验
