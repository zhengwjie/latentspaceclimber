
from tkinter import W
from VAE import VAE,loss_function,loss_function_AE,AE,decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from vae_utils import load_G_D
from utils1 import make_noise
from torch.utils.data import TensorDataset,DataLoader
from vae_utils import samples_to_image
def train_VAE(generator,discriminator,epoch=200):
    # build model
    # get datasets
    # train the model
    z_dim=generator.dim_z
    vae = VAE(x_dim=z_dim, h_dim1= 128, h_dim2=128, z_dim=128)
    # 要对数据进行归一化
    # 
    samples=make_noise(6000,z_dim)
    sample_min=torch.min(samples)
    sample_max=torch.max(samples)
    samples=(samples-sample_min)/(sample_max-sample_min)
    # print(torch.min(samples))
    # print(torch.max(samples))
    train_dataset=TensorDataset(samples)
    train_dataloader=DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    # loss_function
    # 数据集已经准备好了
    if torch.cuda.is_available():
        vae.cuda()
    vae.train()
    optimizer = optim.Adam(vae.parameters())
    
    for step in range(epoch):
        train_loss = 0
        for i,(data,) in enumerate(train_dataloader):
            data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, log_var,z = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)  
            std = torch.exp(0.5*log_var)                                                                                                                                                                                                                                                                                                                                                                 
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(train_loss)
    torch.save(vae.state_dict(),'./weights/vae_gaussian_.pth')
    return vae
def train_AE(generator,discriminator,epoch=200):
    # build model
    # get datasets
    # train the model
    z_dim=generator.dim_z
    ae = AE(x_dim=z_dim, h_dim1= 64, h_dim2=32, z_dim=2)
    # 要对数据进行归一化
    # 
    samples=make_noise(6000,z_dim)
    sample_min=torch.min(samples)
    sample_max=torch.max(samples)
    samples=(samples-sample_min)/(sample_max-sample_min)
    # print(torch.min(samples))
    # print(torch.max(samples))
    train_dataset=TensorDataset(samples)
    train_dataloader=DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    # loss_function
    # 数据集已经准备好了
    if torch.cuda.is_available():
        ae.cuda()
    ae.train()
    optimizer = optim.Adam(ae.parameters(),lr=0.01)
    for step in range(epoch):
        train_loss = 0
        for i,(data,) in enumerate(train_dataloader):
            data = data.cuda()
            optimizer.zero_grad()
            recon_batch, z = ae(data)
            loss = loss_function_AE(recon_batch, data)                                                                                                                                                                                                                                                                                                                                                                 
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(train_loss)
    torch.save(ae.state_dict(),'./weights/ae_gaussian_hidden_2.pth')
    return ae
def train_decoder(generator,discriminator,epoch=200):
    generator.eval()
    discriminator.eval()
    in_dim=2
    out_dim=generator.dim_z
    decoder_instance=decoder(in_dim,32,64,out_dim=out_dim)
    samples=torch.rand((50000,in_dim))
    samples=(samples*60)-30.0
    train_dataset=TensorDataset(samples)
    train_dataloader=DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    # loss_function
    # 数据集已经准备好了
    if torch.cuda.is_available():
        decoder_instance.cuda()
    decoder_instance.train()
    criterion=nn.BCELoss()
    optimizer = optim.Adam(decoder_instance.parameters())
    
    for step in range(epoch):
        train_loss = 0
        for i,(data,) in enumerate(train_dataloader):
            data = data.cuda()
            optimizer.zero_grad()
            out_sample=decoder_instance(data)
            out_label=discriminator(generator(out_sample))
            # 要求让输出的图片越真实越好
            real_label=1
            label = torch.FloatTensor(data.size(0)).cuda()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            #errG = torch.mean(F.softplus(-output))
            loss = criterion(out_label, labelv)
            # loss = loss_function_AE(recon_batch, data)                                                                                                                                                                                                                                                                                                                                                                 
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(train_loss)
    torch.save(decoder_instance.state_dict(),'./weights/decoder_gaussian.pth')
    return decoder_instance

# 测试fid,原来的和生成的图片
# 另一方面，要确定潜空间的大小

def test_VAE(vae,generator):
    with torch.no_grad():
        samples=make_noise(6000,vae.x_dim).cuda()
        samples_pics=generator(samples)
        samples_to_image(samples_pics,"data/mnist_samples")
        sample_min=torch.min(samples)
        sample_max=torch.max(samples)
        samples_input=(samples-sample_min)/(sample_max-sample_min)
        vae_out,mu, log_var,z=vae(samples_input)
        print(z)
        # print()
        vae_out=vae_out*(sample_max-sample_min)+sample_min
        reconstruct_samples=generator(vae_out)
        samples_to_image(reconstruct_samples,"data/mnist_reconstruct_samples")

def test_AE(ae,generator):
    with torch.no_grad():
        samples=make_noise(6000,ae.x_dim).cuda()
        samples_pics=generator(samples)
        samples_to_image(samples_pics,"data/mnist_samples")
        sample_min=torch.min(samples)
        sample_max=torch.max(samples)
        samples_input=(samples-sample_min)/(sample_max-sample_min)
        vae_out,z=ae(samples_input)
        print(torch.min(z))
        print(torch.max(z))
        vae_out=vae_out*(sample_max-sample_min)+sample_min
        reconstruct_samples=generator(vae_out)
        samples_to_image(reconstruct_samples,"data/mnist_ae_reconstruct_samples_hidden_2")

def test_decoder(decoder,generator):
    with torch.no_grad():
        samples=make_noise(6000,2).cuda()
        samples=samples*60.0-30.0
        vae_out=decoder(samples)
        print(vae_out)
        reconstruct_samples=generator(vae_out)
        samples_to_image(reconstruct_samples,"data/mnist_reconstruct_samples")
def test_WGAN(wgan):
        with torch.no_grad():
            samples=(torch.randn(6000,2,1)*50.0).cuda()
            vae_out=wgan(samples)
            vae_out=vae_out.squeeze()
            reconstruct_samples=generator(vae_out)
            samples_to_image(reconstruct_samples,"data/mnist_reconstruct_samples")


# 现在这种训练方式存在一定的问题
# 重构出来的图片是一样的
# 使用ae  隐藏层是128,中间表征也是128的时候，得到的图片和重构图片之间的fid是20左右
# 降到2维再重构之后得到的图片的fid是74.9
# 太大了


if __name__=='__main__':
    # generator,discriminator=load_G_D('cifar10')
    generator,discriminator=load_G_D('mnist')
    generator.cuda()
    discriminator.cuda()

    z_dim=generator.dim_z
    # vae = VAE(x_dim=z_dim, h_dim1= 512, h_dim2=256, z_dim=2).cuda()
    # vae.load_state_dict(torch.load('./vae_gaussian.pth'))

    # vae=train_VAE(generator,discriminator,epoch=20000)
    # test_VAE(vae=vae,generator=generator)

    # ae=train_AE(generator,discriminator,epoch=20000)
    # test_AE(ae=ae,generator=generator)

    # decoder_instance=train_decoder(generator,discriminator,epoch=20000)
    # # decoder_instance=decoder(2,32,64,out_dim=128).cuda()
    # # decoder_instance.load_state_dict(torch.load('./weights/decoder_gaussian.pth'))
    # test_decoder(decoder=decoder_instance)
    from generate_normal.WGAN import Generator
    wgan=Generator().cuda()
    wgan.load_state_dict(torch.load('./generate_normal/generator.pt'))
    test_WGAN(wgan)



