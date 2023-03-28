'''
Author: zhengwanjie 495036225@qq.com
Date: 2022-10-27 07:26:16
LastEditors: zhengwanjie 495036225@qq.com
LastEditTime: 2022-11-15 12:52:39
FilePath: /GANLatentDiscovery-master/common_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# 使用svd进行分解
# 要求matrix是一个numpy类型的数组
import numpy as np
import torch
import tensorflow as tf

def svd_Decompose(matrix):
    # u, s, vh=torch.svd(matrix)
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    # s是特征值，vh是特征向量

    return u,s,vh

def get_one_feature_index(feature_value_vector,row_number=9):
    s,vh=feature_value_vector
    s=s[:10]
    sum=np.sum(s)
    p=s/sum
    index = np.random.choice(len(s),size=2,replace=False,p = p.ravel())
    return index

def clamp_z(z,model_name):
    if model_name=='mnist':
        return torch.clamp(z,-3,3)
    elif model_name=='cifar10':
        return torch.clamp(z,-3,3)
    elif model_name=="molecular_drug":
        return tf.clip_by_value(z,-1,1)
    else:
        return torch.clamp(z,-3,3)



def sample_shift(feature_value_vector,sample_range,alphas=None,sacle=None,dim=10,model_name="cifar10"):
    '''
    description: 
    return {*}
    '''
    
    s,vh=feature_value_vector

    if alphas is None:
        alphas=np.random.uniform(-1*sample_range,sample_range,dim)
    else:
        alphas=alphas*sacle

    shift=alphas[0]*vh[0]
    for i in range(1,dim):
        shift=shift+alphas[i]*vh[i]
    
    if sacle is None:
        sacle=1
    # sample_range=sample_range*sacle
    if model_name=='housegan':
        sample_range=sample_range*sacle
        return shift,alphas
    
    shift=sample_range*shift/np.linalg.norm(shift)

    return shift,alphas

def get_shift_for_graph(G,z,radius):
    shift=torch.randn_like(z)

    shift=radius*shift
    # dim=G.dim_z
    # choices=torch.randint(0,2,size=(dim,))
    # for i in range(dim):
    #     choice=choices[i]
    #     if choice==0:
    #         shift[0][i]=-radius
    #     else:
    #         shift[0][i]=radius
    return shift

def sample_shift_axis(feature_value_vector,sample_range,alphas=None,sacle=None,dim=10):
    '''
    description: 
    return {*}
    '''
    
    s,vh=feature_value_vector
    
    if alphas is None:
        alphas=np.random.uniform(-sample_range,sample_range,1)
    else:
        alphas=alphas*sacle
    s=s[:dim]
    prob=s/np.sum(s)
    arr=[i for i in range(dim)]
    index=np.random.choice(a=arr,size=1,p=prob)
    shift=alphas[0]*vh[index]

    return shift,alphas


def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
def get_child_vector(parent_vector,feature_value_vector,alpha,model_name,domain_range):
    _,vh=feature_value_vector
    shift=alpha[0]*vh[0]
    for i in range(1,len(alpha)):
        shift=shift+alpha[i]*vh[i]
        
    if model_name=="molecular_drug":
        shift=tf.Variable(shift)
    else:
        shift=torch.tensor(shift).to(device=device)
    if model_name=='house_design':
        shift=shift.reshape(10,128)
    if model_name=="house_design" or model_name=="molecular_drug":
        z_plus_shift=parent_vector+shift
        z_plus_shift=clamp_z(z_plus_shift,model_name)
        return z_plus_shift
    
    shift=domain_range*shift/np.linalg.norm(shift.cpu())
    z_plus_shift=parent_vector+shift

    z_plus_shift=clamp_z(z_plus_shift,model_name)
    
    return z_plus_shift

def compute_dim(feature_value_vector,contribute_rate):
    i=0
    s,vh=feature_value_vector
    current_rate=0
    sum_feature_values=np.sum(s)
    while current_rate<contribute_rate:
        if s[i]/sum_feature_values<0.05:
            break
        current_rate=round(current_rate+s[i]/sum_feature_values,2)
        i=i+1
    return i

def sample_group(feature_value_vector,domain_range,dim,group_number):
    shifts=[]
    for i in range(group_number):
        shift,_ = sample_shift(feature_value_vector,domain_range,dim=dim)
        shifts.append(shift)
    return shifts

def get_shift(z,domain_range):
    shift=torch.zeros_like(z)
    high=z.shape[1]
    shift_dims=torch.randint(0,high,(10,))
    shift_values = domain_range*(2*torch.rand(10)-1)
    for i in range(len(shift_dims)):
        shift_dim=shift_dims[i]
        shift_val=shift_values[i]
        shift[0][shift_dim]=shift_val
    return shift
def get_shift_housegan(z,domain_range):
    shift=torch.zeros_like(z)
    high=z.shape[1]
    shift_dims=torch.randint(0,high,(10,10))
    shift_values = domain_range*(2*torch.rand(10,10)-1)
    for i in range(len(shift_dims)):
        for j in range(len(shift_dims[i])):
            shift_dim=shift_dims[i][j]
            shift_val=shift_values[i][j]
            shift[0][i][shift_dim]=shift_val
    return shift

stylegan2_ada_dict={
    "afhqcat":"models/pretrained/generators/stylegan2-ada/afhqcat.pkl",
    "afhqdog":"models/pretrained/generators/stylegan2-ada/afhqdog.pkl",
    "afhqwild":"models/pretrained/generators/stylegan2-ada/afhqwild.pkl",
    "brecahad":"models/pretrained/generators/stylegan2-ada/brecahad.pkl",
    "ffhq":"models/pretrained/generators/stylegan2-ada/ffhq.pkl",
    "metfaces":"models/pretrained/generators/stylegan2-ada/metfaces.pkl",
    "cifar10":"models/pretrained/generators/stylegan2-ada/cifar10u-cifar-ada-best-fid.pkl"
}
def load_stylegan(model_name):
    import pickle
    import sys
    sys.path.append("/home/zhengwanjie/GANLatentDiscovery-master/models/stylegan2_ada/")
    # from models.stylegan2_ada import dnnlib
    # from models.stylegan2_ada import legacy
    # with dnnlib.util.open_url(stylegan2_ada_dict.get(model_name)) as f:
    #     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    with open(stylegan2_ada_dict.get(model_name), 'rb') as f:
        models=pickle.load(f)
        G = models['G_ema']
        D= models["D"]
    return G.eval().cuda()

def make_noise_stylegan(G):
    z=torch.randn(1,G.z_dim).to(device)
    return z


@torch.no_grad()
def random_sample_stylegan(G,z,domain_range=1,num=81):
    # 要把步长的信息保留下来
    # 要计算原点的相对位置
    # 
    # all_alphas=[]
    same_seed(0)
    zs=[]
    # generate_mks=G(z_plus_shift,given_y,given_w)
        
    # img=draw_image(generate_mks=generate_mks,real_nodes=real_nodes)
    shifts=[]
    for i in range(num):
        shift = domain_range*(2*torch.rand(1,G.z_dim)-1)
        radius=np.random.rand(1)[0]*domain_range

        shift=radius*shift/torch.linalg.norm(shift)
        shift = torch.tensor(shift).to(device=device)
        z_plus_shift = z + shift
        # z_plus_shift=clamp_z(z_plus_shift,model_name)
        shifts.append(shift)
        zs.append(z_plus_shift)
    final_imgs=[]
    k=0
    for i in range(len(zs)//10):
        z_batch=zs[i*10:i*10+10]
        k=i*10+10
        batchz=torch.cat(z_batch,dim=0)
        label=torch.zeros([len(batchz), G.c_dim], device=device)
        batch_imgs=G(batchz,label)
        final_imgs.append(batch_imgs)
    if len(zs)%10:
        z_batch=zs[k:]
        batchz=torch.cat(z_batch,dim=0)
        label=torch.zeros([len(batchz), G.c_dim], device=device)
        batch_imgs=G(batchz,label)
        final_imgs.append(batch_imgs)
    final_imgs=torch.cat(final_imgs,dim=0)

    return final_imgs,zs



def random_sample_stylegan_feature_space(G,z,label,domain_range=1,num=81,seed=0,model_name="stylegan"):
    # 求梯度
    # svd分解
    # 采样
    from DirectionDiscovery.draw_graph import get_feature_value_vector
    same_seed(seed)

    feature_value_vector=get_feature_value_vector(G,z,model_name=model_name,label=label)
    dim=compute_dim(feature_value_vector,0.8)
    same_seed(0)
    zs=[]
    for i in range(num):
        shift,alphas = sample_shift(feature_value_vector,domain_range,dim=dim)
        k=torch.rand(1)[0]
        shift=k*shift
        shift = torch.tensor(shift).to(device=device)

        z_plus_shift = z + shift
        z_plus_shift=clamp_z(z_plus_shift,model_name=None)
        zs.append(z_plus_shift)
    final_imgs=[]
    k=0
    for i in range(len(zs)//10):
        z_batch=zs[i*10:i*10+10]
        k=i*10+10
        batchz=torch.cat(z_batch,dim=0)
        label=torch.zeros([len(batchz), G.c_dim], device=device)
        batch_imgs=G(batchz,label)
        final_imgs.append(batch_imgs)
    if len(zs)%10:
        z_batch=zs[k:]
        batchz=torch.cat(z_batch,dim=0)
        label=torch.zeros([len(batchz), G.c_dim], device=device)
        batch_imgs=G(batchz,label)
        final_imgs.append(batch_imgs)

    img=G(z,label)
    final_imgs[len(final_imgs)-1]=img
    final_imgs=torch.cat(final_imgs,dim=0)
    zs[-1]=z
    return final_imgs,zs


@torch.no_grad()
def random_sample_SN_MNIST(G,z,domain_range=1,num=81):
    same_seed(0)
    zs=[]
    shifts=[]
    for i in range(num):
        shift = domain_range*(2*torch.rand(1,G.dim_z)-1)
        radius=np.random.rand(1)[0]*domain_range

        shift=radius*shift/torch.linalg.norm(shift)
        shift = torch.tensor(shift).to(device=device)
        z_plus_shift = z + shift
        if i==0:
            z_plus_shift = z

        shifts.append(shift)
        zs.append(z_plus_shift)
    final_imgs=[]
    k=0
    for i in range(len(zs)//10):
        z_batch=zs[i*10:i*10+10]
        k=i*10+10
        batchz=torch.cat(z_batch,dim=0)
        batch_imgs=G(batchz)
        final_imgs.append(batch_imgs)
    if len(zs)%10:
        z_batch=zs[k:]
        batchz=torch.cat(z_batch,dim=0)

        batch_imgs=G(batchz)
        final_imgs.append(batch_imgs)
    final_imgs=torch.cat(final_imgs,dim=0)

    return final_imgs,zs


def random_sample_SN_MNIST_feature_space(G,z,domain_range=1,num=81,seed=0):
    from DirectionDiscovery.draw_graph import get_feature_value_vector
    # same_seed(seed)

    feature_value_vector=get_feature_value_vector(G,z)
    dim=compute_dim(feature_value_vector,0.8)
    same_seed(0)
    zs=[]
    for i in range(num):
        shift,alphas = sample_shift(feature_value_vector,domain_range,dim=dim)
        k=torch.rand(1)[0]
        shift=k*shift
        shift = torch.tensor(shift).to(device=device)

        z_plus_shift = z + shift
        if i==0:
            z_plus_shift = z
        z_plus_shift=clamp_z(z_plus_shift,model_name=None)
        zs.append(z_plus_shift)
    final_imgs=[]
    k=0
    for i in range(len(zs)//10):
        z_batch=zs[i*10:i*10+10]
        k=i*10+10
        batchz=torch.cat(z_batch,dim=0)
        batch_imgs=G(batchz)
        final_imgs.append(batch_imgs)
    
    if len(zs)%10:
        z_batch=zs[k:]
        batchz=torch.cat(z_batch,dim=0)
        batch_imgs=G(batchz)
        final_imgs.append(batch_imgs)
    final_imgs=torch.cat(final_imgs,dim=0)
    return final_imgs,zs
