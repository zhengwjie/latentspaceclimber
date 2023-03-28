
# 这里是用来计算novelty的工具

from curses import flash
from email.mime import image
from enum import Flag
from pyexpat import model
import torchvision.models as models
# 加载  wide resnet 50
from torchvision import datasets, transforms
import torch
from sklearn.neighbors import NearestNeighbors
import joblib
import numpy as np
import os,sys
from housegan import housegan_utils


o_path = "/home/zhengwanjie/GANLatentDiscovery-master/"

sys.path.append(o_path)
os.chdir(o_path)
from torch_tools.visualization import to_image
# 明天的任务
from DirectionDiscovery.vae_utils import load_G_D
from utils1 import make_noise
# 找模型 使用resnet18抽特征
# 测距离 
# 算novelty score
# 把novelty score 集成到系统中
# clap(0,1)
from collections import OrderedDict
outputs = []

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_min_novelty=3.1495796996523566

mnist_max_novelty=12.205233742456718

cifar10_min_novelty=4.116129991453906
cifar10_max_novelty=16.961748421221763

@torch.no_grad()
def load_model():
    # wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
    # return wide_resnet50_2
    resnet18=models.resnet18(pretrained=True)
    resnet18.layer1[-1].register_forward_hook(hook)
    resnet18.layer2[-1].register_forward_hook(hook)
    resnet18.layer3[-1].register_forward_hook(hook)
    resnet18.avgpool.register_forward_hook(hook)
    return resnet18.to(device=device).eval()

@torch.no_grad()
def hook(module, input, output):
    # print(output.shape)
    # new_output=output.squeeze()
    new_output=output
    # outputs=torch.cat((outputs,new_output),dim=0)
    outputs.append(new_output)


# 开始抽取特征
@torch.no_grad()
def load_dataset_mnist(feature_extractor):
    global outputs
    train_features = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    dataset_root="data/datasets/minist"
    dataset = datasets.MNIST(
        root=dataset_root,
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]),
        download=True)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    resize=transforms.Resize((224,224))
    final_data=torch.ones(0,1,28,28).to(device)
    outputs=[]
    for batch_idx, (data, _) in enumerate(dataset_loader):
        data=data.to(device)
        
        # data2=data.reshape(100,-1).to(device)
        final_data=torch.cat((final_data,data),dim=0)
        resize_data=resize(data)
        resize_data=torch.cat((resize_data,resize_data,resize_data),dim=1).to(device)
        feature_extractor(resize_data)
        for k, v in zip(train_features.keys(), outputs):
            v=v.cpu()
            train_features[k].append(v)
        outputs=[]
    for k, v in train_features.items():
        train_features[k] = torch.cat(v, 0)
    return final_data.cpu(),train_features['avgpool']
# 加载cifar10的数据集
@torch.no_grad()
def load_dataset_cifar10(feature_extractor):
    global outputs
    train_features = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    dataset_root="data/datasets/cifar10"
    dataset = datasets.CIFAR10(
        root=dataset_root,
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        download=True)
    dataset_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False)
    resize=transforms.Resize((224,224))
    final_data=torch.ones(0,3,32,32).to(device)
    outputs=[]
    for batch_idx, (data, _) in enumerate(dataset_loader):
        print(data.shape)
        data=data.to(device)
        # data2=data.reshape(100,-1).to(device)
        final_data=torch.cat((final_data,data),dim=0)
        # 3*224*224
        resize_data=resize(data).to(device)
        feature_extractor(resize_data)
        for k, v in zip(train_features.keys(), outputs):
            v=v.cpu()
            train_features[k].append(v)
        outputs=[]
    for k, v in train_features.items():
        train_features[k] = torch.cat(v, 0)
    return final_data.cpu(),train_features['avgpool']

def knn(train_X,test_X,n_neighbors=3,target_file="mnist_knn_3.pkl"):
    train_X=train_X.cpu().numpy()
    test_X=test_X.cpu().numpy()
    print(test_X[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(train_X)
    joblib.dump(nbrs,target_file)

    distances, indices = nbrs.kneighbors(test_X)
    print(distances)
    print(indices)
def load_knn(target_file="mnist_knn_3.pkl"):
    return joblib.load(target_file)
def load_dataset(target_file="data/mnist/mnist_knn_dataset.pkl"):
    return joblib.load(target_file)
def train(n_neighbors=15,
    model_name="mnist",
    dataset_target_file="data/mnist/mnist_knn_dataset.pkl",
    knn_feature_nbrs="mnist_knn_15_feature.pkl"
    ):
    feature_extractor=load_model()
    if model_name=='mnist':
        final_data,features=load_dataset_mnist(feature_extractor)
    elif model_name=='cifar10':
        final_data,features=load_dataset_cifar10(feature_extractor)
    
    dataset=final_data,features
    joblib.dump(dataset,dataset_target_file)

    print(final_data.shape)
    # print(features.shape)
    # knn(final_data,final_data[:10],
    # target_file="mnist_knn_15_image.pkl",
    # n_neighbors=n_neighbors)
    
    knn(features.flatten(1),features[:10].flatten(1),
    target_file=knn_feature_nbrs,
    n_neighbors=n_neighbors
    )
def test_knn(nbrs,test_X):
    distances, indices = nbrs.kneighbors(test_X)
    print(distances)
    print(indices)
def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix
# 通过传入参数，得到图像的novelty评分

@torch.no_grad()
def novelty_map_pixel_level(images,model_name,feature_extractor,nbrs,dataset):
    from scipy.ndimage import gaussian_filter
    import torch.nn.functional as F
    from tqdm import tqdm
    n_neighbors=2
    test_outputs=transform_images(images,feature_extractor,model_name)

    # select K nearest neighbor and take average
    # topk_values, topk_indexes = torch.topk(dist_matrix, k=n_neighbors, dim=1, largest=False)
    features=test_outputs['avgpool'].flatten(1).cpu().numpy()

    distances, indices = nbrs.kneighbors(features)

    # indices是二维的数组
    train_images=[]
    for i in range(len(indices)):
        for j in range(n_neighbors):
            train_images.append(dataset.__getitem__(indices[i][j])[0].unsqueeze(0))
    train_images=torch.cat(train_images,0).to(device)
    train_outputs=transform_images(train_images,feature_extractor,model_name)

    score_map_list = []
    # test_outputs['avgpool'].shape[0]是同一类数据的测试样本数量
    image_neighbors=[]
    for t_idx in tqdm(range(test_outputs['avgpool'].shape[0])):
        score_maps = []
        for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer
            print(layer_name)
            # construct a gallery of features at all pixel locations of the K nearest neighbors
            topk_feat_map = train_outputs[layer_name][t_idx*n_neighbors:(t_idx+1)*n_neighbors]
            test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
            feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)
          
            # calculate distance matrix
            dist_matrix_list = []
            for d_idx in range(feat_gallery.shape[0] // 100):
                dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                dist_matrix_list.append(dist_matrix)
            dist_matrix = torch.cat(dist_matrix_list, 0)

            # k nearest features from the gallery (k=1)
            score_map = torch.min(dist_matrix, dim=0)[0]
            score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=32,
                                        mode='bilinear', align_corners=False)
            score_maps.append(score_map)

        image_neighbors.append(train_images[t_idx*n_neighbors:(t_idx+1)*n_neighbors])

        # average distance between the features
        score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

        # apply gaussian smoothing on the score map
        score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
        score_map_list.append(score_map)
    return score_map_list,image_neighbors

@torch.no_grad()
def compute_pixel_level_novelty(images,model_name,feature_extractor,nbrs,dataset):
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm
    if(torch.max(images)-torch.min(images))>1.5:
        images=(images+1)/2
    n_neighbors=1
    test_outputs=transform_images(images,feature_extractor,model_name)

    # select K nearest neighbor and take average
    # topk_values, topk_indexes = torch.topk(dist_matrix, k=n_neighbors, dim=1, largest=False)
    features=test_outputs['avgpool'].flatten(1).cpu().numpy()

    distances, indices = nbrs.kneighbors(features)

    # indices是二维的数组
    train_images=[]
    for i in range(len(indices)):
        for j in range(n_neighbors):
            train_images.append(dataset.__getitem__(indices[i][j])[0].unsqueeze(0))
    train_images=torch.cat(train_images,0).to(device)

    score_map_list = []
    # test_outputs['avgpool'].shape[0]是同一类数据的测试样本数量
    resize=transforms.Resize((32,32))
    neighber_images=[]
    for i in range(len(images)):
        # 1*32*32或者3*32*32的
        test_image=resize(images[i])
        neighber_image=resize(train_images[i])
        neighber_image_PIL=to_image(neighber_image,True)
        neighber_images.append(neighber_image_PIL)
        neighber_image_PIL.save("neighboor.png")
        
        score_map=torch.mean(abs(test_image-neighber_image),dim=0)
        score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=2)
        score_map=score_map.squeeze().cpu().detach().numpy()
        score_map_list.append(score_map)
    return score_map_list,neighber_images
# 
# 默认取得的邻居数量是3
@torch.no_grad()
def get_neighbor_images(images,model_name,feature_extractor,nbrs,dataset,n_neighbors=3):
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm
    if(torch.max(images)-torch.min(images))>1.5:
        images=(images+1)/2
    print(images.shape)
    test_outputs=transform_images(images,feature_extractor,model_name)

    # select K nearest neighbor and take average
    # topk_values, topk_indexes = torch.topk(dist_matrix, k=n_neighbors, dim=1, largest=False)
    features=test_outputs['avgpool'].flatten(1).cpu().numpy()
    print(features.shape)
    distances, indices = nbrs.kneighbors(features)

    neighber_images=[]
    for i in range(len(indices)):
        for j in range(n_neighbors):
            img=dataset.__getitem__(indices[i][j])[0].unsqueeze(0)
            neighber_image_PIL=to_image(img,True)
            neighber_images.append(neighber_image_PIL)

    return neighber_images
def test(knn_model_image,knn_model_feature):

    nbrs_image=load_knn(knn_model_image)
    nbrs_feature=load_knn(knn_model_feature)
    dataset=load_dataset()
    final_data,features=dataset
    features=features['avgpool']

    test_X_image=final_data[:10].cpu().numpy()
    test_X_feature=features[:10].cpu().numpy()
    distances_image, indices_image = nbrs_image.kneighbors(test_X_image)
    save_image("data/mnist/mnist_knn_image",indices_image,final_data)
    print(distances_image)
    print(indices_image)
    distances_feature, indices_feature = nbrs_feature.kneighbors(test_X_feature)
    save_image("data/mnist/mnist_knn_feature",indices_feature,final_data)
    print(distances_feature)
    print(indices_feature)
    return
# indices是一个二维数组
def save_image(root,indices,image_dataset):
    import os
    if not os.path.exists(root):
        os.mkdir(root)
    if(len(image_dataset.shape)==2):
        image_dataset=image_dataset.reshape(-1,1,28,28)
    for i in range(len(indices)):
        new_dir=os.path.join(root,str(i))
        if not os.path.exists(new_dir):
           os.mkdir(new_dir)
        for j in range(len(indices[i])):
            file_name=os.path.join(new_dir,str(j)+".jpg")
            indice=indices[i][j]
            img=to_image(image_dataset[indice],True)
            img.save(file_name)
    return

def transform_images(images,feature_extractor,model_name):
    # images 的范围在[-1,1]之间，需要把它转为[0,1]
    if  (torch.max(images)-torch.min(images))>1.5:
        images=(images+1)/2
    global outputs
    if model_name=='mnist':
        transform=transforms.Compose([
        # transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((224,224))
    ])
    elif model_name=='cifar10':
        transform=transforms.Compose([
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((224,224))
    ])
    images=transform(images)
    if(len(images.shape)==3):
        images=images.unsqueeze(0)
    if model_name=='mnist' and len(images.shape)==4:
        images=torch.cat((images,images,images),dim=1)
    outputs=[]
    feature_extractor(images)
    intermediate_features = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    for k, v in zip(intermediate_features.keys(), outputs):
        v=v.cpu()
        intermediate_features[k].append(v)
    for k, v in intermediate_features.items():
        intermediate_features[k] = torch.cat(v, 0)
        # print(intermediate_features[k].shape)
    outputs=[]
    return intermediate_features


# feature_extractor
@torch.no_grad()
def novelty_score_min_max(images,feature_extractor,model_name,nbrs):
    intermediate_features=transform_images(images,feature_extractor,model_name)
    features=intermediate_features['avgpool'].flatten(1).cpu().numpy()
    distances, indices = nbrs.kneighbors(features)
    print(np.mean(distances,axis=1))
    return np.min(np.mean(distances,axis=1)),np.max(np.mean(distances,axis=1))

def image_to_base64(img):
    import base64
    from io import BytesIO
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    binary_data = output_buffer.getvalue()
    base64_data = str(base64.b64encode(binary_data), encoding="utf-8")
    return base64_data
# feature_extractor
@torch.no_grad()
def novelty_score(images,feature_extractor,model_name,nbrs,norm=True):

    intermediate_features=transform_images(images,feature_extractor,model_name)
    
    features=intermediate_features['avgpool'].flatten(1).cpu().numpy()

    distances, indices = nbrs.kneighbors(features)

    score=np.mean(distances,axis=1)
    if norm:
        if(model_name=='mnist'):
            score=(score-mnist_min_novelty)/(mnist_max_novelty-mnist_min_novelty)
        elif model_name=='cifar10':
            score=(score-cifar10_min_novelty)/(cifar10_max_novelty-cifar10_min_novelty)
        return np.clip(score,0,1)
    else:
        return score
    # return the images
    # 可以根据后台的逻辑判断是不是需要返回邻居的图片

    

def test_knn_mnist():
    # load the generator and produce images
    generator,_=load_G_D("mnist")
    generator.to(device)
    min_val=10000
    max_val=0      
    nbrs=joblib.load("mnist_knn_15_feature.pkl")
    feature_extractor=load_model()
    model_name="mnist"
    for i in range(100):
        z=make_noise(100,generator.dim_z).to(device=device)
        images=generator(z)
        min_number,max_number=novelty_score_min_max(images,feature_extractor,model_name,nbrs)
        min_val=min(min_number,min_val)
        max_val=max(max_number,max_val)
    print(min_val)
    print(max_val)
def test_knn_cifar10():
    # load the generator and produce images
    generator,_=load_G_D("cifar10")
    generator.to(device)
    min_val=10000
    max_val=0      
    nbrs=joblib.load("cifar10_knn_15_feature.pkl")
    feature_extractor=load_model()
    model_name="cifar10"
    for i in range(50):
        z=make_noise(100,generator.dim_z).to(device=device)
        images=generator(z)
        min_number,max_number=novelty_score_min_max(images,feature_extractor,model_name,nbrs)
        min_val=min(min_number,min_val)
        max_val=max(max_number,max_val)
    print(min_val)
    print(max_val)
def load_knn_dataset(model_name):
    print(model_name)
    dataset=None
    if model_name=='SN_MNIST' or model_name=="mnist":
        dataset=load_dataset("data/mnist/mnist_knn_dataset.pkl")
    elif model_name=='stylegan2-ada_cifar10' or model_name=="cifar10":
        dataset=load_dataset(target_file="data/cifar10/cifar10_knn_dataset.pkl")
    return dataset
def load_image_dataset(model_name):
    dataset=None
    if model_name=='SN_MNIST' or model_name=="mnist":
        dataset_root="data/datasets/mnist"
        dataset = datasets.MNIST(
        root=dataset_root,
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=False)
    elif model_name=='stylegan2-ada_cifar10' or model_name=="cifar10":
        dataset_root="data/datasets/cifar10"
        dataset = datasets.CIFAR10(
        root=dataset_root,
        train=True, 
        transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        download=False)
    elif model_name=='molecular_drug':
        from Drug_Generator.model_use import drug_generator
        return drug_generator.load_drug_dataset()
    else:
        return housegan_utils.load_hougan_node_10()
    return dataset

def load_knn_model(model_name):
    model=None
    if model_name=='SN_MNIST' or model_name=="mnist":
        model=load_knn("data/mnist/mnist_knn_15_feature.pkl")
    elif model_name=='stylegan2-ada_cifar10' or model_name=="cifar10":
        model=load_knn("data/cifar10/cifar10_knn_15_feature.pkl")
    elif model_name=='molecular_drug':
        model=load_knn("data/drug/drug_knn_15.pkl")
    else:
        model=load_knn("data/housegan/housegan_knn_15.pkl")
    return model


if __name__=='__main__':

    test_knn_cifar10()
    # test_knn_mnist()
    # train(n_neighbors=15,
    # model_name="cifar10",
    # dataset_target_file="data/cifar10/cifar10_knn_dataset.pkl",
    # knn_feature_nbrs="cifar10_knn_15_feature.pkl")
    

    # test_knn()

    # train(n_neighbors=15,
    # model_name="mnist",
    # dataset_target_file="data/mnist/mnist_knn_dataset.pkl",
    # knn_feature_nbrs="mnist_knn_15_feature.pkl")



    # train(n_neighbors=15)
    # test(knn_model="mnist_knn_image.pkl")
    # test(knn_model="mnist_knn_feature.pkl")
    # test(knn_model_image="mnist_knn_image.pkl",knn_model_feature="mnist_knn_feature.pkl")

    # final_data=final_data.numpy()
    
    # nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(final_data)
    # joblib.dump(nbrs,"mnist_knn_3.pkl")
    # import time
    # current_time=time.time()
    # test_data=final_data[:10,:]
    # distances, indices = nbrs.kneighbors(test_data)
    # final_time=time.time()-current_time
    # print(final_time)
    # print(distances)
    # print(indices)


# 使用knn算法

# 计算novelty分数


