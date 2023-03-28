import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)

def load_G_D(model_name):
    from loading import load_G_D_CIFAR10,load_G_D_MNIST,load_G_D_gaussian
    if model_name=='cifar10':
        G,D=load_G_D_CIFAR10()
    elif model_name=='mnist':
        G,D=load_G_D_MNIST()
    elif model_name=='gaussian':
        G,D=load_G_D_gaussian()
    return G.eval(),D.eval()

def samples_to_image(samples,dataset_target_path,start_index=0,target_size=28):
    import os
    o_path = os.getcwd()
    import sys
    sys.path.append(o_path)
    if not os.path.exists(dataset_target_path):
        os.mkdir(dataset_target_path)
    from torch_tools.visualization import to_image
    for i in range(len(samples)):
        img=to_image(samples[i],True)
        img=img.resize((target_size,target_size))
        # print(dataset_target_path+"/"+str(i+start_index)+".png")
        print(i+start_index)
        img.save(dataset_target_path+"/"+str(i+start_index)+".png")
