
from operator import mod
import os
from pyexpat import model
import re
from signal import pthread_kill
import sys
import math

from tkinter.messagebox import NO
from traceback import print_tb
import numpy as np
import time
from scipy.fftpack import shift

from DirectionDiscovery import novelty
o_path = os.getcwd()

sys.path.append(o_path)

import torch
from matplotlib import image, pyplot as plt
from torchvision.utils import make_grid
import torch.autograd.functional as F
from functools import partial
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
import resnet_model
import joblib
from scipy import stats
# define some constants
from DirectionDiscovery.vae_utils import load_G_D
from Drug_Generator.model_use import drug_generator
from common_utils import clamp_z,svd_Decompose,sample_shift,same_seed
import math
import tensorflow as tf
from housegan import housegan_utils
deformator_dirs={
    "SN_MNIST":"./models/pretrained/deformators/SN_MNIST/",
    "SN_Anime":"./models/pretrained/deformators/SN_Anime",
    "stylegan2-ada_cifar10":"./models/pretrained/deformators/stylegan2-ada/",
    "BigGAN_ImageNet":"./models/pretrained/deformators/BigGAN/",
    "ProgGAN_CelebA_HQ":"./models/pretrained/deformators/ProgGAN/",
    "stylegan2_ffhq":"./models/pretrained/deformators/StyleGAN2"
}
GAN_weights={
    'BigGAN_ImageNet': 'models/pretrained/generators/BigGAN/G_ema.pth',
    'ProgGAN_CelebA_HQ': 'models/pretrained/generators/ProgGAN/100_celeb_hq_network-snapshot-010403.pth',
    'SN_MNIST': 'models/pretrained/generators/SN_MNIST',
    'SN_Anime': 'models/pretrained/generators/SN_Anime',
    'stylegan2_ffhq': 'models/pretrained/StyleGAN2/stylegan2-car-config-f.pt',
    'stylegan2-ada_cifar10':'models/pretrained/generators/stylegan2-ada/cifar10u-cifar-ada-best-fid.pkl'
}

rob_predictors={
    "mnist":(rob_predictor.Rob_predictor,"./weights/rob_predictor/rob_predictor_mnist.pt"),
    'cifar10':(rob_predictor.Rob_predictor,"./weights/rob_predictor/rob_predictor_resnet56.pt")
}
classifiers={
    # "Lenet1": (lenet.Lenet1,"./weights/mnist/mnist_lenet1.pt"),
    # "Lenet4": (lenet.Lenet4,"./weights/mnist/mnist_lenet4.pt"),
    "mnist": (lenet.Lenet5,"./weights/mnist/mnist_lenet5.pt"),
    'cifar10':(resnet_model.cifar10_resnet56,"./weights/cifar10/cifar10_resnet56.pt")
}



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

@torch.no_grad()
def load_generator(generator_name,bias=True):

    deformator_dir=deformator_dirs.get(generator_name)
    G_weights=GAN_weights.get(generator_name)
    deformator, G, shift_predictor = load_from_dir(
            deformator_dir,
            G_weights=G_weights,
            bias=bias)
    return G.to(device),deformator.to(device)
@torch.no_grad()
def load_generator_descriminator(model_name):
    if model_name=='SN_MNIST' or model_name=='mnist':
        generator,discriminator=load_G_D('mnist')
    elif model_name=='stylegan2-ada_cifar10' or model_name=='cifar10':
        generator,discriminator=load_G_D('cifar10')
    elif model_name=='drug_generator' or model_name=='molecular_drug':
        generator=drug_generator.load_drug_generator()
        return generator,None
    else:
        generator=housegan_utils.load_housegan()
        return generator,None

    return generator.to(device),discriminator.to(device)

# 模型在cuda上
@torch.no_grad()
def load_UMAP_neural(in_dim,middle_dim,out_dim,path='./DirectionDiscovery/train/trained_model/UMAP_mnist_model_50000.pth'):
    model=UMAP_Dense_Network(in_dim,middle_dim,out_dim)
    model=model.to(device=device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

@torch.no_grad()
def generate_image(G,z):
    if z is None:
        z=make_noise(1, G.dim_z).to(device=device)
    img = G(z).cpu()[0]
    return img
@torch.no_grad()
def generate_shifted_image(G,deformator,z,deformator_input):
    if deformator is not None:
        latent_shift = deformator(deformator_input)
    else:
        latent_shift = deformator_input
    shifted_image = G.gen_shifted(z, latent_shift).cpu()[0]
    return shifted_image
@torch.no_grad()
def generate_plane(G,deformator,directions,z=None,shifts_r=8.0,shifts_count=8):
    if z is None:
        z=make_noise(1, G.dim_z).to(device=device)
    images=[]
    list_range=range(-shifts_count,shifts_count+1,1)
    for i in list_range:
        row_imgs=[]
        for j in list_range:
            if i*i+j*j==0:
                img=generate_image(G,z)
                # if(len(img.shape)==4):
                #     img=img.squeeze(0)
                row_imgs.append(img)
                continue
            deformator_input=one_hot(deformator.input_dim, i/shifts_count*shifts_r, directions[0]).to(device=device) + one_hot(deformator.input_dim, j/shifts_count*shifts_r, directions[1]).to(device=device)
            shifted_image=generate_shifted_image(G,deformator,z,deformator_input)
            # if(len(shifted_image.shape)==4):
            #     shifted_image=shifted_image.squeeze(0)

            row_imgs.append(shifted_image)
        images.append(row_imgs)
    return images
# 使用umap产生图片
# 输入是在[-8,8]
@torch.no_grad()
def generate_plane_umap(G,deformator,umap_neural,directions,z=None,shifts_count=20,shifts_range=8):
    if z is None:
        z=make_noise(1, G.dim_z).to(device=device)
    images=[]
    list_range=range(-shifts_count,shifts_count+1,1)
    for i in list_range:
        row_imgs=[]
        for j in list_range:
            input_array=[[i*(shifts_range/shifts_count),j*(shifts_range/shifts_count)]]
            input_data=torch.tensor(input_array).float().to(device=device)
            out_data=umap_neural(input_data).squeeze()
            # 把out_data转化为shift的格式
            deformator_input=one_hot(deformator.input_dim, 0, 0).to(device=device)
            for k in range(len(directions)):
                deformator_input[directions[k]]=out_data[k]
            shifted_image=generate_shifted_image(G,deformator,z,deformator_input)
            # if(len(shifted_image.shape)==4):
            #     shifted_image=shifted_image.squeeze(0)

            row_imgs.append(shifted_image)
        images.append(row_imgs)
    return images

# 只有一个分类器的路径不够
@torch.no_grad()
def load_classifier(classifier_name):
    classifier_specified=classifiers.get(classifier_name)
    classifier=classifier_specified[0]().to(device=device)
    # Load the pretrained model
    classifier.load_state_dict(torch.load(classifier_specified[1]))
    classifier.eval()
    return classifier


@torch.no_grad()
def load_rob_predictor(rob_predictor_name):

    rob_predictor_specified=rob_predictors.get(rob_predictor_name)
    rob_predictor_model=rob_predictor_specified[0]().to(device=device)
    rob_predictor_model.load_state_dict(torch.load(rob_predictor_specified[1]))
    rob_predictor_model.eval()
    
    return rob_predictor_model
    # load the rob_predictor

# resize=Resize((28,28))
# 
# mnist : 0-0.31
# cifar10: 0.004
@torch.no_grad()
def get_robustness_value(classifier,rob_predictor_model,img,normalize=None,resize=None):
    if resize is not None:
        img=resize(img)
    if normalize is not None:
        img=normalize(img)

    if(len(img.shape)==3):
        img=img.unsqueeze(0)
        
    penultimate_layer = rob_predictor.get_penultimate_layer(classifier, img.to(device))
    robustness_predict = rob_predictor_model(torch.tensor(penultimate_layer).to(device))
    return robustness_predict

@torch.no_grad()
def get_robustness_plane_values(classifier,rob_predictor_model,imgs,normalize=None,resize=None):
    robustness_values=[]
    print(imgs)
    for i in range(len(imgs)):
        img=imgs[i]
        robustness_value=get_robustness_value(classifier,rob_predictor_model,img,normalize,resize)
        robustness_values.append(robustness_value.item())
    return robustness_values

@torch.no_grad()
def plane_to_image(plane_images,file_path):
    rows_count=len(plane_images)
    fig, axs = plt.subplots(rows_count,figsize=(20, 20))

    for ax, row_imgs in zip(axs, plane_images):
        ax.axis('off')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.1, hspace=0.1)
        ax.imshow(to_image(make_grid(row_imgs, nrow=rows_count, padding=1), True))
    fig.canvas.draw()
    plt.close(fig)
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imgs=[]
    # crop borders
    nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
    img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
    imgs.append(img)
    print('saving chart to {}'.format(file_path))
    Image.fromarray(np.hstack(imgs)).save(file_path)

@torch.no_grad()
def get_robustness_values(model_name,directions,z=None):
    G,deformator=load_generator(model_name)
    rob_predictor_model=load_rob_predictor(model_name)
    # classifier name
    classifier_name="Lenet1"
    classifier=load_classifier(classifier_name=classifier_name)
    
    umap_neural=load_UMAP_neural(2,40,5)

    if z is None:
        z=make_noise(1, G.dim_z).to(device=device)
    shifts_count=20
    # plane_images=generate_plane(G,deformator,directions,z,shifts_count=shifts_count)
      # get the robustness value
    
    plane_images=generate_plane_umap(G,deformator,umap_neural,directions,z,shifts_count=shifts_count,shifts_range=8)
    normalize=Normalize(-1,2)
    resize=Resize((28,28))

    robustness_values=get_robustness_plane_values(classifier,rob_predictor_model,plane_images,normalize,resize)
    
    # 这里plane_images没有经过归一化和resize
    return robustness_values,plane_images

# 默认在坐标轴上取20个点
def get_axis_in_plane(direction,shifts_range=8,shifts_count=20):
    input_data=np.zeros((2*shifts_count+1,5))
    for i in range(-shifts_count,shifts_count+1):
        input_data[i,direction]=i*shifts_range/shifts_count
    # 数据构建好了
    # 加载模型
    umap_model=load_umap(model_path='./DirectionDiscovery/train/trained_model/umap.model')
    # 预测数据
    axis_data=umap_model.transform(input_data)
    # 获取二维空间的数据，把它转换为-8到+8之间
    # (41,2)
    # 
    dataset=torch.load("./DirectionDiscovery/train/trained_model/dataset_50000.pt")
    x=dataset[:][0]
    print(torch.min(x))
    print(torch.max(x))
    axis_data=(axis_data-torch.min(x).numpy())/(torch.max(x)-torch.min(x)).numpy()
    # 转化到(-8,8)
    # (41,2)
    return axis_data.tolist()
# 计算GAN的jacobian矩阵
# 要求G和z处在同一个设备上
def jacobian_G(G,z,label=None):
    from utils1 import is_conditional
    if label is not None:
        func, params, buffers= make_functional_with_buffers(G)
        jacobian_GAN=F.jacobian(partial(func,params,buffers,c=label),z)
    elif is_conditional(G):
        func, params, buffers= make_functional_with_buffers(G.big_gan)
        y=G.big_gan.shared(G.mixed_classes(1).to(device))
        jacobian_GAN=F.jacobian(partial(func,params,buffers,y=y),z)
    else:
        func, params, buffers= make_functional_with_buffers(G)
        jacobian_GAN=F.jacobian(partial(func,params,buffers),z)
        
    # 所以目前求出来的矩阵结果是对的
    # 初步估计
    # print(jacobian_G.sum())
    jacobian_GAN=jacobian_GAN.squeeze()

    jacobian_GAN=jacobian_GAN.reshape(-1,jacobian_GAN.shape[-1])
    return jacobian_GAN


def get_feature_value_vector(G,z,model_name=None,given_y=None,given_w=None,label=None):
    from common_utils import svd_Decompose
    # use different method according to different model
    if model_name=="molecular_drug":
        return drug_generator.get_drug_feature_vector(G,z)
    if model_name=='house_design' or model_name=="housegan":
        return housegan_utils.get_housegan_feature_vector(G,z,given_y,given_w)
    if model_name=='stylegan':
        jacobian_GAN=jacobian_G(G,z,label=label)
    else:
        jacobian_GAN=jacobian_G(G,z)
    jacobian_GAN=jacobian_GAN.cpu().detach().numpy()
    # jacobian_GAN=jacobian_GAN.cpu().detach()

    u,s,vh=svd_Decompose(jacobian_GAN)

    return s,vh
def get_min_max_probability_dense(z_dim):
    
    min_probability_dense=stats.multivariate_normal.pdf(
        mean=torch.zeros(z_dim),
        cov=torch.ones(z_dim),
        x=1.3*torch.ones(z_dim))
    max_probability_dense=stats.multivariate_normal.pdf(
        mean=torch.zeros(z_dim),
        cov=torch.ones(z_dim),
        x=0.8*torch.ones(z_dim))
    return math.log(min_probability_dense),math.log(max_probability_dense)    



def show(imgs,file_name="result3.png"):
    import torchvision.transforms.functional as F
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    fig, axs = plt.subplots(len(imgs))
    for i,row_imgs in enumerate(imgs):
        axs[i].axis('off')
        axs[i].imshow(to_image(make_grid(row_imgs, nrow=len(row_imgs), padding=1), True))
    plt.savefig(file_name)
def draw_svd_feature_vector(G,z,start_shift=-5,end_shift=5,count=15):
    jacobian_GAN=jacobian_G(G,z)
    jacobian_GAN=jacobian_GAN.cpu().detach().numpy()

    u,s,vh=svd_Decompose(jacobian_GAN)
    shift_r=end_shift-start_shift
    images=[]
    for i in range(10):
        row_images=[]
        ans=torch.tensor(vh[i])
        # print(torch.mean(ans))
        # print(torch.std(ans))
        for shift in np.arange(start_shift, end_shift + 1e-9, shift_r / count):
            shift_vector=shift*vh[i]
            shift_vector=torch.tensor(shift_vector).to(device=device)
            print(torch.mean(z+shift_vector))
            print(torch.std(z+shift_vector))
            test_vector=(z).detach().cpu()
            test_vector=(z+shift_vector).detach().cpu()
            u=torch.mean(test_vector)
            std=torch.std(test_vector)
            print(stats.kstest(test_vector,'norm',(torch.zeros_like(test_vector),torch.ones_like(test_vector))))
            img=G(z+shift_vector)[0]
            # img=to_image(img,True)
            row_images.append(img)
        images.append(row_images)
    return images

# 进行回归
# 然后产生一系列的图片

def regression(feature_vector,deformator,G,z):

    dirs=tuple(deformator.parameters())[0]
    X = dirs.cpu().detach().numpy()
    y=feature_vector
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)
    arrs=[(X[:,i],reg.coef_[i]) for i in range(len(reg.coef_))]
    sorted_arrs=sorted(arrs,key=lambda x:abs(x[1]),reverse=True)
    shifted_imgs=[]
    shifted_imgs.append(to_image(G(z)[0]))
    shifted_imgs.append(to_image(G(z+torch.tensor(feature_vector).to(device=device))[0]))
    for i in range(len(sorted_arrs)):
        shifted_vector=torch.tensor(sorted_arrs[i][1]*sorted_arrs[i][0]).to(device=device)
        img=G(z+shifted_vector)[0]
        shifted_imgs.append(to_image(G(z)[0]))
        shifted_imgs.append(to_image(img))
    return shifted_imgs

    
# jacobian
# svd分解
def svd_jacobian(G,z,feature_vector):
    # det_z
    # det_img
    img=G(z).squeeze().reshape(-1)
    shift=torch.tensor(torch.tensor(feature_vector).to(device=device))
    shifted_img=G(z+shift).squeeze().reshape(-1)
    det_img=shifted_img-img
    final_result=torch.zeros(0,len(det_img)).to(device)
    for i in range(len(shift)):
        det=(det_img/shift[i]).unsqueeze(0)
        print(det.shape)
        print(final_result.shape)
        final_result=torch.cat([final_result,det],dim=0)
        # 3072
    # final_result.shape=[3072,512]
    jacobian=final_result.cpu().detach().numpy().T

    u,s,vh=svd_Decompose(jacobian)
    # 返回一系列的图片
    print(vh.shape)
    shifted_imgs=[]
    shifted_imgs.append(to_image(G(z)[0]))
    shifted_imgs.append(to_image(G(z+shift)[0]))
    for i in range(len(s)):
        shifted_vector=torch.tensor(s[i]*vh[i]).to(device=device)
        print(s[i])
        img=G(z+shifted_vector)[0]
        shifted_imgs.append(to_image(G(z)[0]))
        shifted_imgs.append(to_image(img))
    # print(reg.coef_)
    return shifted_imgs

# sefa获取方向，然后进行回归
def sefa_regression(feature_vector,G,z):
    from utils1 import factorize_weight
    layers, boundaries, values = factorize_weight(G)
    # boundaries是特征向量，values是特征值

    X = boundaries[0:30].T
    y=feature_vector
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X, y)
    print(reg.coef_)
    print(reg.score(X,y))
    arrs=[(X[:,i],reg.coef_[i]) for i in range(len(reg.coef_))]
    sorted_arrs=sorted(arrs,key=lambda x:abs(x[1]),reverse=True)
    shifted_imgs=[]
    shifted_imgs.append(to_image(G(z)[0]))
    shifted_imgs.append(to_image(G(z+torch.tensor(feature_vector).to(device=device))[0]))
    for i in range(len(sorted_arrs)):
        shifted_vector=torch.tensor(sorted_arrs[i][1]*sorted_arrs[i][0]).to(device=device)
        img=G(z+shifted_vector)[0]
        shifted_imgs.append(to_image(G(z)[0]))
        shifted_imgs.append(to_image(img))
    return shifted_imgs

# 要求必须要有feature_value_vector
def get_feature_indexs(feature_value_vector,row_number=9):
    s,vh=feature_value_vector
    s=s[:10]
    sum=np.sum(s)
    p=s/sum
    index = np.random.choice(len(s),size=2,replace=False,p = p.ravel())
    feature_indexs=[]
    for i in range(row_number):
        for j in range(row_number):
            # 选两个向量
            index = np.random.choice(len(s),size=2,replace=False,p = p.ravel())
            # 小的index放在y轴，大的index放在x轴
            min_index=np.min(index)
            max_index=np.max(index)
            feature_indexs.append([min_index,max_index])
    return feature_indexs


# 构建平面
# 以原点为中心，以随机的方式，构建一个平面
# 以特征值作为概率是否合适
# 不单单要返回图片，同时要返回两个特征值
# 
def normalize_prob(prob,model_name):
    if prob==0:
        return 0
    if model_name=="mnist":
        prob=(math.log(prob)-(-230))/(-160+230)
    # TO DO:
    #   记录一下 cifar10的最大值和最小值
    #   
    elif model_name=='cifar10':
        prob=(math.log(prob)-(-544.4400719213812))/(-536.083737868877+544.4400719213812)
    if prob<0:
        prob=0
    if prob>1:
        prob=1
    return prob



@torch.no_grad()
def generate_svd_plane(G,z,domain_range,
feature_value_vector=None,shifts_count=9,discriminator=None,
given_alphas=None,feature_extractor=None,model_name=None,
knn_model=None,knn_dataset=None,given_y=None,given_w=None,scale=None
):
    if model_name=="molecular_drug":
        return drug_generator.generate_drug_svd_plane(G, z, domain_range,
        feature_value_vector = feature_value_vector, shifts_count = shifts_count, discriminator = discriminator
        ,feature_extractor = feature_extractor, model_name = model_name,
        knn_model = knn_model, knn_dataset = knn_dataset)
    if model_name=='house_design':
        return housegan_utils.generate_housegan_svd_plane(G,z,given_y=given_y,given_w=given_w,
        domain_range=domain_range,
        feature_value_vector = feature_value_vector, shifts_count = shifts_count, discriminator = discriminator,
        given_alphas = given_alphas, feature_extractor = feature_extractor, model_name = model_name,
        knn_model = knn_model, knn_dataset = knn_dataset,scale=scale)
        
    if feature_value_vector is None:
        feature_value_vector=get_feature_value_vector(G,z)

    from scipy import stats
    # 拆分domain_x 与domain_y

    mean=torch.zeros(z.shape[1])
    cov=torch.ones(z.shape[1])

    # 构建一个以概率取向量的过程
    images=[]
    # 要把步长的信息保留下来
    # 要计算原点的相对位置
    # 
    all_alphas=[]
    same_seed(0)
    zs=[]
    index=0
    start_range=-(shifts_count//2)
    end_range=(shifts_count//2)+1
    collected_imgs=[]
    for i in range(start_range,end_range):
        for j in range(start_range,end_range):
            alpha=None
            if given_alphas is not None:
                alpha=given_alphas[index]
            
            shift,alphas = sample_shift(feature_value_vector,domain_range,alpha,scale)
            shift=torch.tensor(shift).to(device=device)
            
            z_plus_shift=z+shift
            if i==0 and j==0:
                z_plus_shift=z
                alphas=np.zeros_like(alphas)

            all_alphas.append(alphas)

            z_plus_shift=clamp_z(z_plus_shift,model_name)
            
            probability=stats.multivariate_normal.pdf(
                mean=mean,
                cov=cov,
                x=z_plus_shift.cpu()
            )
            probability=normalize_prob(probability,model_name)
            zs.append(z_plus_shift)
            img=G(z_plus_shift)
            collected_imgs.append(img)

            realness=0
            # 在执行模型的时候，需要把z_plus_shift去掉
            images.append([None,str(probability),str(realness),None,None])
            index=index+1
    final_z=torch.cat(zs,dim=0)
    import pickle
    with open("final_z.pkl","wb") as f:
        pickle.dump(final_z,f)
    with open("z.pkl","wb") as f:
        pickle.dump(z,f)

    final_imgs=torch.cat(collected_imgs,dim=0)
    from DirectionDiscovery import novelty
    # knn_images先不做处理
    score=novelty.novelty_score(final_imgs,feature_extractor,model_name,knn_model)
    if discriminator:
        realnesses=discriminator(final_imgs)
    # 计算鲁棒性的值
    robust_values=get_robust(final_imgs,model_name)
    print(robust_values)
    robust_values=torch.tensor(robust_values)

    print(robust_values)
    if model_name=="mnist":
        robust_values=robust_values/0.32
    else:
        robust_values=robust_values/0.004
        # robust_values=torch.clamp(robust_values,0.0001,2)
        # robust_values=(torch.log(robust_values)-math.log(0.001))/(math.log(0.01)-math.log(0.001))
        robust_values=torch.clamp(robust_values,0,1)

    for i in range(len(final_imgs)):
        img=to_image(final_imgs[i],True)
        images[i][0]=img
        if discriminator:
            images[i][2]=realnesses[i].item()
        images[i][3]=score[i]
        images[i][4]=robust_values[i].item()

    return images,all_alphas

@torch.no_grad()
def generate_svd_plane_rs(G,z,domain_range,
feature_value_vector=None,shifts_count=9,discriminator=None,
given_alphas=None,feature_extractor=None,model_name=None,
knn_model=None,knn_dataset=None,given_y=None,given_w=None,scale=None
):


    from scipy import stats
    # 拆分domain_x 与domain_y

    mean=torch.zeros(z.shape[1])
    cov=torch.ones(z.shape[1])

    # 构建一个以概率取向量的过程
    images=[]
    # 要把步长的信息保留下来
    # 要计算原点的相对位置
    # 
    all_alphas=[]
    same_seed(0)
    zs=[]
    index=0
    start_range=-(shifts_count//2)
    end_range=(shifts_count//2)+1
    collected_imgs=[]
    for i in range(start_range,end_range):
        for j in range(start_range,end_range):
            
            # shift = domain_range*(2*torch.rand(1,G.z_dim)-1)
            shift = domain_range*(2*torch.rand(1,G.dim_z)-1)
            
            
            radius=np.random.rand(1)[0]*domain_range

            shift=radius*shift/torch.linalg.norm(shift)
            shift = torch.tensor(shift).to(device=device)
            
            z_plus_shift=z+shift
            if i==0 and j==0:
                z_plus_shift=z

            z_plus_shift=clamp_z(z_plus_shift,model_name)
            
            probability=stats.multivariate_normal.pdf(
                mean=mean,
                cov=cov,
                x=z_plus_shift.cpu()
            )
            probability=normalize_prob(probability,model_name)
            zs.append(z_plus_shift)
            img=G(z_plus_shift)
            collected_imgs.append(img)

            realness=0
            # 在执行模型的时候，需要把z_plus_shift去掉
            images.append([None,str(probability),str(realness),None,None])
            index=index+1
    final_z=torch.cat(zs,dim=0)
    

    final_imgs=torch.cat(collected_imgs,dim=0)
    from DirectionDiscovery import novelty
    # knn_images先不做处理
    score=novelty.novelty_score(final_imgs,feature_extractor,model_name,knn_model)
    if discriminator:
        realnesses=discriminator(final_imgs)
    # 计算鲁棒性的值
    robust_values=get_robust(final_imgs,model_name)
    print(robust_values)
    robust_values=torch.tensor(robust_values)

    print(robust_values)
    if model_name=="mnist":
        robust_values=robust_values/0.32
    else:
        robust_values=robust_values/0.004
        # robust_values=torch.clamp(robust_values,0.0001,2)
        # robust_values=(torch.log(robust_values)-math.log(0.001))/(math.log(0.01)-math.log(0.001))
        robust_values=torch.clamp(robust_values,0,1)

    for i in range(len(final_imgs)):
        img=to_image(final_imgs[i],True)
        images[i][0]=img
        if discriminator:
            images[i][2]=realnesses[i].item()
        images[i][3]=score[i]
        images[i][4]=robust_values[i].item()

    return images,final_z



def compute_rate(s,number=10,rates=0.8):
    sum_feature_values=np.sum(s)
    feature_value_rate=[]
    accumulating_contribute_rates=[]
    current_rate=0
    i=0
    while current_rate<rates:
        feature_value_rate.append(s[i]/sum_feature_values)
        current_rate=current_rate+s[i]/sum_feature_values
        ++i
    print(feature_value_rate)
     
    add_value=0
    
    if len(feature_value_rate)>11:
        add_value=sum(feature_value_rate[11:])
    print(add_value)
    
    feature_value_rate[0]=feature_value_rate[0]+add_value
    feature_value_rate=feature_value_rate[0:11]
    accumulate=0
    for i in range(len(feature_value_rate)):
        accumulate=accumulate+feature_value_rate[i]
        if accumulate>1:
            accumulate=1
        feature_value_rate[i]=str(feature_value_rate[i])
        accumulating_contribute_rates.append(str(round(accumulate,2)))

    return feature_value_rate,accumulating_contribute_rates

def get_points(score_map):
    length=len(score_map)
    points=[]
    threshold=np.percentile(score_map,80)
    for i in range(length):
        for j in range(length):
            if(score_map[i][j]>threshold):
                points.append({
                    "x":str((i/length)*100),
                    "y":str((j/length)*100),
                    "value":str(score_map[i][j]),
                    "radius":str(100/32),
                })
    print(points)
    return points

@torch.no_grad()
def get_image_info(z,generator,model_name,feature_value_vector=None,shift=None,
    given_y=None,given_w=None,rates=0.9):
    import time
    if shift is not None:
        if model_name=="molecular_drug":
            shift=tf.Variable(shift)
        else:
            shift=torch.tensor(shift).to(device=device)
        if model_name=='house_design':
            shift=shift.reshape(10,128)
        z_plus_shift=z+shift
        z_plus_shift=clamp_z(z_plus_shift,model_name)
    else:
        z_plus_shift=z
    if feature_value_vector is None:
        feature_value_vector=get_feature_value_vector(generator,z_plus_shift,given_y=given_y,given_w=given_w,model_name=model_name)
    s,vh=feature_value_vector
    feature_value_rate,accumulating_contribute_rates=compute_rate(s,rates=rates)

    return feature_value_rate,accumulating_contribute_rates
def get_nextz(global_z,shift_z,model_name):
    if model_name=="molecular_drug":
        shift_z=tf.Variable(shift_z)
    else:
        shift_z=torch.tensor(shift_z).to(device=device)
    if model_name=="house_design":
        shift_z=shift_z.reshape(10,128)
    z_plus_shift=global_z+shift_z
    z_plus_shift=clamp_z(z_plus_shift,model_name)

    return z_plus_shift
# imgs是[b,c,w,h]
def get_robust(imgs,model_name):
    classifier=load_classifier(model_name)
    rob_predictor=load_rob_predictor(model_name)
    normalize=Normalize(-1,2)
    # normalize=None
    resize=Resize((28,28))

    robust_values=get_robustness_plane_values(classifier,rob_predictor,imgs,normalize=normalize,resize=resize)
    return robust_values

    

# 把获取邻居的过程准备好
def get_neighbors(z,generator,model_name,dataset,nbrs,
feature_extractor,feature_value_vector=None,shift=None,
given_y=None,given_w=None):
    if shift is not None:
        if model_name=="molecular_drug":
            shift=tf.Variable(shift)
        else:
            shift=torch.tensor(shift).to(device=device)
        if model_name=="house_design":
            shift=shift.reshape(10,128)
        z_plus_shift=z+shift
        z_plus_shift=clamp_z(z_plus_shift,model_name)
    else:
        z_plus_shift=z
    
    if feature_value_vector is None:
        feature_value_vector=get_feature_value_vector(generator,z_plus_shift,model_name=model_name,given_y=given_y,given_w=given_w)
    if model_name=="molecular_drug":
        latent_vector=generator.generator(z_plus_shift)
        return drug_generator.get_neighbor_moleculars(latent_vector,feature_extractor,nbrs,dataset)
    if model_name=='house_design':
        gene_masks=generator(z_plus_shift,given_y,given_w)
        return housegan_utils.get_neighbor_designs(gene_masks,feature_extractor,nbrs,dataset)
    images=generator(z_plus_shift)
    neighbor_images=novelty.get_neighbor_images(
        images,model_name,feature_extractor,nbrs,dataset
    )
    return neighbor_images


def get_mask_image(raw_image,score_map):
    import numpy
    if  (torch.max(raw_image)-torch.min(raw_image))>1.5:
        raw_image=(raw_image+1)/2
    raw_image=raw_image.cpu().detach().numpy()
    raw_image=raw_image.transpose(1, 2, 0)
    threshold=np.percentile(score_map,75)
    score_map[score_map > threshold] = 1
    score_map[score_map < threshold] = 0
    mask_image = raw_image.copy()
    print(numpy.min(raw_image))
    mask_image[score_map == 0] = numpy.min(raw_image)

    return to_image(torch.tensor(mask_image.transpose(2,0,1)),True)

def getheatmap_image(raw_image,score_map):
    import numpy
    if  (torch.max(raw_image)-torch.min(raw_image))>1.5:
        raw_image=(raw_image+1)/2
    img=to_image(raw_image,True)
    img=img.convert("RGBA")
    x,y=img.size
    min_val=np.min(score_map)
    max_val=np.max(score_map)
    for i in range(x):
        for j in range(y):
            color=img.getpixel((i,j))
            val=score_map[i][j]
            val=int(255*(val-min_val)/(max_val-min_val))
            color=color[:-1]+(val,)
            img.putpixel((i,j),color)
    return img

def generate_init_noise(model_name,generator,given_y=None):
    if model_name=='mnist' or model_name=='cifar10':
        from common_utils import same_seed
        same_seed(0)
        # return torch.randn(1,generator.dim_z).to(device=device)
        
        # return 1.3*torch.ones(1,generator.dim_z).to(device=device)
        # z=torch.randn(1,generator.dim_z).to(device=device)

        
        z=make_noise(30,generator.dim_z)[26:27].to(device=device)
        

        # return torch.mean(z,dim=0).unsqueeze(0).to(device=device)
        return z

    elif model_name=='molecular_drug':
        return drug_generator.get_init_noise(gan=generator)
    else:
        return housegan_utils.get_init_noise(given_nodes=given_y)



def generate_noise(model_name,generator,given_y=None,noise_number=1):
    if model_name=='mnist' or model_name=='cifar10':
        return make_noise(noise_number,generator.dim_z).to(device=device)
    elif model_name=='molecular_drug':
        return drug_generator.get_valid_noise(gan=generator,sample_number=noise_number)
    else:
        return housegan_utils.get_noise(given_y,noise_number)
# 还需要降维，所以需要数据的特征


def generate_img(model_name,generator,z,given_y,given_w,feature_extractor):
    img=None
    feature=None
    if model_name=="molecular_drug":
        img,feature=drug_generator.generate_image(generator,z)
        feature=tf.reshape(feature,(1,-1))
    elif model_name=='house_design':
        img,feature=housegan_utils.generate_image(generator,z,given_y,given_w)
        feature=feature.reshape(1,-1)
    else:
        img=generator(z)
        img1=img
        feature=novelty.transform_images(img,feature_extractor,model_name)
        feature=feature['avgpool'].reshape(1,-1)
        img=to_image(img1)
    return img,feature
    

def generate_samples(model_name,generator,given_y,given_w,feature_extractor=None):
    zs=[]
    imgs=[]
    features=[]
    zs=generate_noise(model_name,generator,given_y,35)
    for i in range(35):
        z=zs[i:i+1]
        img,feature=generate_img(model_name,generator,z,given_y,given_w,feature_extractor)
        imgs.append(img)
        features.append(feature)
    # for i in range(len(history_z)):
    #     z=history_z[i].to(device)
    #     img,feature=generate_img(model_name,generator,z,given_y,given_w,feature_extractor)
    #     zs.append(z)
    #     imgs.append(img)
    #     features.append(feature)
    # 把降维结果也返回
    if model_name=="molecular_drug":
        features=tf.concat(features,0).numpy()
    else:
        features=torch.concat(features,dim=0).cpu().detach().numpy()
    from sklearn.manifold import TSNE
    
    ts = TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(features)

    return imgs,zs,result








        
