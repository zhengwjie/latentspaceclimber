import sys
import os
import numpy as np
from umap import UMAP
o_path = os.getcwd()

sys.path.append(o_path)
from DirectionDiscovery.draw_graph import load_generator_descriminator
from utils1 import make_noise
import torch
from DirectionDiscovery.novelty import load_model,transform_images,load_knn_model,novelty_score
from sklearn.neighbors import NearestNeighbors
# 首先加载GAN和准备采样数据
# 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
import joblib
from torch_tools.visualization import to_image
from umap import UMAP
from sklearn.manifold import TSNE

novelty_weights={
    "mnist":"data/mnist/novelty_overview_dataset.pkl",
    "cifar10":"data/cifar10/novelty_overview_dataset.pkl",
    "SN_MNIST":"data/mnist/novelty_overview_dataset.pkl",
    "stylegan2-ada_cifar10": "data/cifar10/novelty_overview_dataset.pkl"
}
umap_save_model={
    "mnist": "data/mnist/umap_overview_model.pkl",
    "cifar10": "data/cifar10/umap_overview_model.pkl",
    "SN_MNIST": "data/mnist/umap_overview_model.pkl",
    "stylegan2-ada_cifar10": "data/cifar10/umap_overview_model.pkl"
}
overview_dataset={
    "mnist": "data/mnist/overview_dataset_gridc.pkl",
    "cifar10": "data/cifar10/overview_dataset.pkl",
    "SN_MNIST": "data/mnist/overview_dataset_grida.pkl",
    "stylegan2-ada_cifar10": "data/cifar10/overview_dataset.pkl"
}

@torch.no_grad()
def novelty_overview(model_name,sample_number,feature_extractor):
    generator, _ = load_generator_descriminator(model_name=model_name)
    all_z = []
    all_images = []
    features = []
    novelty_scores = []
    feature_nbrs = load_knn_model(model_name)
    for i in range(sample_number // 500):
        z = make_noise(500, generator.dim_z).to(device=device)
        all_z.append(z.detach().cpu())
        imgs = generator(z)
        all_images.append(imgs.cpu())
        feature = transform_images(imgs, feature_extractor, model_name)
        features.append(feature['avgpool'].squeeze())
        # print(feature['avgpool'].shape)
        novelty_score_batch = novelty_score(imgs, feature_extractor, model_name, feature_nbrs, norm=False)
        # print(novelty_score_batch)
        novelty_scores.append(novelty_score_batch)
        # print(type(novelty_score_batch))
    if sample_number % 500:
        z = make_noise(sample_number % 500, generator.dim_z).to(device=device)
        all_z.append(z.detach().cpu())
        imgs = generator(z)
        all_images.append(imgs.cpu())
        feature = transform_images(imgs, feature_extractor, model_name)
        features.append(feature['avgpool'].squeeze())
        novelty_score_batch = novelty_score(imgs, feature_extractor, model_name, feature_nbrs, norm=False)
        # print(novelty_score_batch)
        novelty_scores.append(novelty_score_batch)

    all_z = torch.cat(all_z, dim=0)
    features = torch.cat(features, dim=0)
    # 现在所有的信息都有了，接下来就是把他们进行保存
    # 降维，保存数据
    # 把 features 降维
    # 使用 umap 降维
    # 最后执行一次插值
    
    # umap_instance=UMAP(n_components=2)
    tsne_instance=TSNE(n_components=2)
    features = np.array(features)

    # reduced_features=umap_instance.fit_transform(features)
    reduced_features=tsne_instance.fit_transform(features)

    # 使用reduced_features构建一个k近邻算法
    # 对于一个点来说，根据k近邻算法获取它在高维空间的坐标
    nbrs = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(reduced_features)
    
    # 接下来，可以绘制一个contour的地图
    x_min,x_max=np.min(reduced_features[:,0]),np.max(reduced_features[:,0])
    y_min,y_max=np.min(reduced_features[:,1]),np.max(reduced_features[:,1])


    novelty_overview_dataset=[all_z,nbrs,x_min,x_max,y_min,y_max]
    joblib.dump(novelty_overview_dataset,novelty_weights.get(model_name))
    joblib.dump(tsne_instance,umap_save_model.get(model_name))
    novelty_scores=np.concatenate(novelty_scores)
    print(novelty_scores.shape)
    print(reduced_features.shape)
    get_novelty_2D(reduced_features,novelty_scores)
    
    # 
    # 还需要t sne 降维，所以需要抽取特征
    # 降维，还需要计算novelty
# 给一个二维坐标，根据二维坐标，计算图片的novlty
# point 是一个二维的网格数据
@torch.no_grad()
def get_novelty(model_name,feature_extractor,row_number=500):
    generator,_=load_generator_descriminator(model_name)
    feature_nbrs=load_knn_model(model_name)
    dataset_path=novelty_weights.get(model_name)
    novelty_overview_dataset=joblib.load(dataset_path)
    nbrs=novelty_overview_dataset[1]
    all_z=novelty_overview_dataset[0]
    x_min,x_max=novelty_overview_dataset[2],novelty_overview_dataset[3]
    y_min,y_max=novelty_overview_dataset[4],novelty_overview_dataset[5]
    points,num_x,num_y=getpoints(x_min,x_max,y_min,y_max,row_number)

    distances, indices = nbrs.kneighbors(points)
    zs=[]
    for j in range(len(distances)):
        distance_sum=np.sum(distances[j])
        z=torch.zeros_like(all_z[0])
        for i in range(len(indices[j])):
            indice=indices[j][i]
            z=z+(distances[j][i]/distance_sum)*all_z[indice]
        z=z.unsqueeze(0).to(device=device)
        zs.append(z)
    zs=torch.cat(zs,dim=0)
    # 分batch size 生成图片
    final_novelty_score=[]
    k=-1
    batch_number=500
    for i in range(len(zs)//batch_number):
        zs_batch=zs[batch_number*i:batch_number*(i+1)]
        k=i
        imgs_batch=generator(zs_batch)
        novelty_score_batch=novelty_score(imgs_batch,feature_extractor,model_name,feature_nbrs,norm=False)
        final_novelty_score.append(novelty_score_batch)
    if(len(zs)%batch_number):
        zs_batch = zs[batch_number * (k+1):]
        imgs_batch = generator(zs_batch)
        novelty_score_batch = novelty_score(imgs_batch, feature_extractor, model_name, feature_nbrs, norm=False)
        final_novelty_score.append(novelty_score_batch)
    return np.concatenate(final_novelty_score).tolist(),num_x,num_y
def get_novelty_2D(reduced_features,novelty_scores,row_number=100):
    import matplotlib.pyplot as plt
    import scipy
    from scipy.interpolate import griddata


    # 接下来，可以绘制一个contour的地图
    x_min,x_max=np.min(reduced_features[:,0]),np.max(reduced_features[:,0])
    y_min,y_max=np.min(reduced_features[:,1]),np.max(reduced_features[:,1])
    grid_x,grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    #generate grid data using the points and values above
    grid_a = griddata(reduced_features, novelty_scores, (grid_x, grid_y), method='cubic')

    grid_b = griddata(reduced_features, novelty_scores, (grid_x, grid_y), method='linear')

    grid_c = griddata(reduced_features, novelty_scores, (grid_x, grid_y), method='nearest')

    #visualizations
    fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(func(grid_x,grid_y))
    axs[0, 0].set_title("main")
    axs[1, 0].imshow(grid_a, extent=(x_min,x_max,y_min,y_max), origin='lower')
    axs[1, 0].set_title("cubic")
    axs[0, 1].imshow(grid_b, extent=(x_min,x_max,y_min,y_max), origin='lower')
    axs[0, 1].set_title("linear")
    axs[1, 1].imshow(grid_c, extent=(x_min,x_max,y_min,y_max), origin='lower')
    axs[1, 1].set_title("nearest")
    fig.tight_layout()
    plt.savefig('graph.png')
    novelty_grida=grid_a.reshape(-1).tolist(),100,100
    novelty_gridb=grid_b.reshape(-1).tolist(),100,100
    novelty_gridc=grid_c.reshape(-1).tolist(),100,100
    joblib.dump(novelty_grida,"data/mnist/overview_dataset_grida.pkl")
    joblib.dump(novelty_gridb,"data/mnist/overview_dataset_gridb.pkl")
    joblib.dump(novelty_gridc,"data/mnist/overview_dataset_gridc.pkl")

    return 


def load_novelty(model_name):
    dataset=joblib.load(overview_dataset.get(model_name))
    return dataset
def getpoints(x_min,x_max,y_min,y_max,row_number):
    range_y=y_max-y_min
    range_x=x_max-x_min
    points=[]
    num_y=len(np.arange(y_min, y_max + 1e-8,  range_y/ (row_number-1)))
    num_x=len(np.arange(x_min, x_max + 1e-8,  range_x/ (row_number-1)))

    for j in np.arange(y_min, y_max + 1e-8,  range_y/ (row_number-1)):
        for i in np.arange(x_min,x_max + 1e-8,range_x/ (row_number-1)):
            points.append([i,j])
    points = np.array(points)

    return points,num_x,num_y

# 把降维的模型放到内存中
def get_umap(model_name):
    try:
        umap_instance=joblib.load(umap_save_model.get(model_name))
        return umap_instance
    except:
        return None

def load_increase_dim_model(model_name):
    try:
        increase_dim_model_instance=joblib.load(novelty_weights.get(model_name))
        return increase_dim_model_instance
    except:
        return None

# here we assume the len of zs is not so large, and can be solved in a time,
# and all the
# return  coordinate_in_2D (k,2) k is the size of the zs
# 这里是为了把探索过的图片展示在概览上展示出来
@torch.no_grad()
def get_coordinate_in_two_dim(zs, generator, feature_extractor, umap_model, model_name):
    imgs=generator(zs)
    intermediate_features = transform_images(imgs, feature_extractor, model_name)
    features = intermediate_features['avgpool'].squeeze().cpu().numpy()
    print(features.shape)
    if len(features.shape)==1:
        features=features.reshape(1,-1)
    coordinate_in_2D=umap_model.transform(features)
    images=[]
    for i in range(len(imgs)):
        images.append(to_image(imgs[i],True))
    return coordinate_in_2D,images

@torch.no_grad()
def increase_dim(point_x,point_y,novelty_overview_dataset,generator):
    point=[[point_x,point_y]]
    nbrs=novelty_overview_dataset[1]
    all_z=novelty_overview_dataset[0]

    distances, indices = nbrs.kneighbors(point)
    distance=distances[0]
    indice=indices[0]
    sum_dis=np.sum(distance)

    z = torch.zeros_like(all_z[0]).to(device)
    for i in range(len(indice)):
        z = z + (distance[i]/ sum_dis) * all_z[indice[i]].to(device)
    z=z.unsqueeze(0)
    img=generator(z)
    img=to_image(img,True)
    return z,img














# 需要回去
# 需要计算
if __name__=='__main__':

    feature_extractor=load_model()
    model_name="mnist"
    # generator, _ = load_generator_descriminator(model_name=model_name)
    # feature_nbrs=load_knn_model(model_name)
    #
    # novelty_score_real=get_novelty(model_name,feature_extractor,generator,feature_nbrs,row_number=50)
    # print(novelty_score_real)

    sample_number=6000

    novelty_overview(model_name,sample_number,feature_extractor)

    mnist_dataset=get_novelty(model_name,feature_extractor,row_number=100)
    joblib.dump(mnist_dataset,"data/mnist/overview_dataset.pkl")

    # model_name = "cifar10"

    # novelty_overview(model_name, sample_number, feature_extractor)

    # cifar10_dataset=get_novelty(model_name,feature_extractor)
    # joblib.dump(cifar10_dataset,overview_dataset.get(model_name))



    # grid_x, grid_y = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
    #
    # from scipy.interpolate import griddata
    # grid_z0 = griddata(reduced_features, novelty_scores, (grid_x, grid_y), method='nearest')
    # grid_z1 = griddata(reduced_features, novelty_scores, (grid_x, grid_y), method='linear')
    # grid_z2 = griddata(reduced_features, novelty_scores, (grid_x, grid_y), method='cubic')
    # print(grid_z2.shape)
    # 500 * 500
    # import matplotlib.pyplot as plt
    # plt.subplot(221)
    # plt.scatter(reduced_features[:,0], reduced_features[:,1], c=novelty_scores)
    # plt.title('Original')
    # plt.subplot(222)
    # plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    # plt.title('Nearest')
    # plt.subplot(223)
    # plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    # plt.title('Linear')
    # plt.subplot(224)
    # plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    # # plt.title('Cubic')
    # plt.gcf().set_size_inches(6, 6)
    # plt.savefig("novelty_overview_cifar10.png")
    # print("saved")


