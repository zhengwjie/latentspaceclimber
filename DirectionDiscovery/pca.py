import imp
from tkinter.messagebox import NO
from tkinter.tix import X_REGION
import numpy as np
import os
import sklearn
from sklearn.decomposition import PCA
import torch
from draw_graph import load_generator
from torch_tools.visualization import to_image
from torchvision.utils import make_grid
from utils1 import make_noise, one_hot
import matplotlib.pyplot as plt
from PIL import Image
# 首先方向有128个方向
# 128*z.dim
# 转化为 64*z.dim
# k*m 的shift
@torch.no_grad()
def reduce_dim(X,n_compents,pca=None):
    X=X.cpu()
    if pca is None:
        pca=PCA(n_components=n_compents)
        pca.fit(X)
    X_r=pca.transform(X).T
    return torch.tensor(X_r,dtype=torch.float32).cuda(),pca
def pca1(XMat, k):
    """
    XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
    k：表示取前k个特征值相应的特征向量
    finalData：指的是返回的低维矩阵
    reconData：相应的是移动坐标轴后的矩阵
    """
    XMat=XMat.cpu().numpy()
    print(XMat)
    average = meanX(XMat) 
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
        # print(selectVec.shape)
        finalData = data_adjust * selectVec.T 
        reconData = (finalData * selectVec) + average  
    return torch.tensor(selectVec+average,dtype=torch.float32).cuda(),\
    torch.tensor(finalData.T,dtype=torch.float32).cuda(),\
    torch.tensor(reconData.T,dtype=torch.float32).cuda()
#计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征    
def meanX(dataX):
    return np.mean(dataX,axis=0)#axis=0表示依照列来求均值。假设输入list,则axis=1

@torch.no_grad()
def get_X(deformator,shift_value):
    total_dim=deformator.input_dim
    concat_input=one_hot(deformator.input_dim, shift_value, 0).cuda().unsqueeze(0)
    for i in range(1,total_dim):
        shift_input=one_hot(deformator.input_dim, shift_value, i).cuda().unsqueeze(0)
        concat_input=torch.concat([concat_input,shift_input],axis=0)
    print(concat_input)
    output=deformator(concat_input).T
    return output

# shifts 的维度是[k,128]
# 返回[k,1,32,32]张图片
@torch.no_grad()
def generate_images(G,z,shifts):
    image=G(z+shifts)
    return image

# 把图片保存下来
@torch.no_grad()
def convert_to_image(images):
    converted_images=[]
    number=images.shape[0]
    for i in range(number):
        img=to_image(images[i])
        converted_images.append(img)
    return converted_images

def images_to_fig(original_img,imgs):
    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count,figsize=(40,40))

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))
    
    texts = range(0,rows_count)
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=len(shifts_imgs), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)
    return fig

def save_to_dir(original_img,imgs,out_dir):
    os.makedirs(out_dir, exist_ok=True)

    step = 20
    max_dim=len(imgs)
    for start in range(0, max_dim - 1, step):
        new_imgs=[]
        current_imgs=imgs[start:start+step]
        fig=images_to_fig(original_img,current_imgs)
        fig.canvas.draw()
        plt.close(fig)
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # crop borders
        nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
        img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
        new_imgs.append(img)

        out_file = os.path.join(out_dir, '{}_{}.jpg'.format(start, start+step))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(new_imgs)).save(out_file)
def generate_all_images(G,deformator,z,shift_count,shift_r=8.0,n_compents=64):
    middle_images=[]


    for shift_value in np.arange(-shift_r, shift_r + 1e-9, shift_r / shift_count):
        X=get_X(deformator,shift_value)

        # X_r,pca=reduce_dim(X,n_compents,pca=pca)
        eigVector,final_, X_r=pca1(X,n_compents)
        # X_r=X
        images=generate_images(G,z,X_r)
        middle_images.append(images)
    final_images=[]
    for row in range(n_compents):
        row_imgs=[]
        for j in range(len(middle_images)):
            row_imgs.append(middle_images[j][row])
        final_images.append(row_imgs)
    return final_images





if __name__ =="__main__":
    rob_predictor_path = "./weights/rob_predictor/rob_predictor.pt"
    classifier_path = "./weights/mnist/mnist_lenet5.pt"
    deformator_dir="./models/pretrained/deformators/SN_MNIST/"
    G_weights='./models/pretrained/generators/SN_MNIST/'
        # load all the model
    G,deformator=load_generator(deformator_dir,G_weights)
    z=make_noise(1, G.dim_z).cuda()
    original_image=G(z).cpu()
    images=generate_all_images(G,deformator,z,3,n_compents=64)
    save_to_dir(original_image,images,"./mnist/reduce_dim5/")


# import numpy as np

# axis=0表示按行求均值，求行的均值
# axis=1表示按列求均值，求列的均值
# np.mean(,axis=0)




        


