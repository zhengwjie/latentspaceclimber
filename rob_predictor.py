# Numpy
from locale import normalize
from statistics import mean
from matplotlib.pyplot import axis, cla
import numpy as np
# utilities
import math
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Torchvision
import torchvision
from torchvision import datasets, transforms
# foolbox
import foolbox as fb
# classifier
# import lenet
# import alexnet
# import resnet
# import vgg
# import mobileNetV2
# import googlenet_huy
# import googLenet
# import shufflenetV2
import lenet
import os
import time


# ====================================
#     定义regression neural network
# ====================================
class Regression_Net(nn.Module):
    def __init__(self):
        super(Regression_Net, self).__init__()
        self.fc1 = nn.Linear(10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# ====================================
#     Defination: robustness predictor
# ====================================
class Rob_predictor(nn.Module):
    def __init__(self):
        super(Rob_predictor, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc4 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# =============================================
# Method：Get the penultimate layer of a single image, return a (10,) list
# =============================================
def get_penultimate_layer(classifier, img_data):

    with torch.no_grad():
        return classifier(img_data).detach().cpu().tolist()



# =============================================
# Method: Get the real robustness value of a single image, return a number
# this method use adversial attack to get the robust value
# =============================================
def get_robustness_value(classifer_fmodel, attack, img_data, img_label, epsilons):
    
    for epsilon in epsilons:
       
        # Call FGSM Attack
        raw, clipped, is_adv = attack(classifer_fmodel, img_data, img_label, epsilons=epsilon)

        if is_adv.item() == True:
            return epsilon

    
    return epsilon


# ======================================================
# Method: Get the dataset composed of penultimate layers(as input) and 
# real robustness values(as label), which are from the correctly classified images.
# ======================================================
def get_dataset_for_rob_predictor(classifier, classifier_fmodel, attack, data_loader, epsilons, trainset_for_predictor_filename,training_number):

    penultimate_layer_list = []
    robustness_value_list = []
    iterator = 0
    maxnum = 0

    # Loop over all examples in test set
    for data, label in data_loader:

        # Send the data and label to the device
        data, label = data.to(device), label.to(device)
        # 对数据进行了正则化
        data_normalized = normalize(data)

        # Forward pass the data through the model
        # output是分类器的输出
        with torch.no_grad():
            output = classifier(data_normalized)

        # get the index of the max log-probability
        init_pred = output.max(1)[1]  

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != label.item():
            continue
        
        penultimate_layer = get_penultimate_layer(classifier, data_normalized)

        robustness_value = get_robustness_value(classifier_fmodel, attack, data, label, epsilons)

        penultimate_layer_list.append(penultimate_layer)
        robustness_value_list.append(robustness_value)

        if robustness_value == 0.5:
            maxnum += 1

        print(f"processing img{iterator}, robustness value is {robustness_value:.3f}  ", maxnum)
        iterator += 1

        # 训练集包含10000条数据
        if iterator == training_number:
            break
    # 构造 tensor 向量
    penultimate_layer_list_tsr = torch.tensor(penultimate_layer_list)
    robustness_value_list_tsr = torch.tensor(robustness_value_list).unsqueeze(1)
    trainset_tsr = torch.utils.data.TensorDataset(penultimate_layer_list_tsr, robustness_value_list_tsr)
    # penultimate_layer_list

    torch.save(trainset_tsr, trainset_for_predictor_filename)
    print("Dataset for rob predictor completed.")


# ====================================
# Method: train rob predicting model
# 前一个方法在构造数据集
# 训练数据集
# ==================================== 
def train_rob_predictor():

    # Initialize the network
    rob_predictor = Rob_predictor().to(device)

    optimizer = optim.Adam(rob_predictor.parameters(), lr=0.0005)
    criterion = nn.MSELoss()


    train_set = torch.load(trainset_for_predictor_PATH)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    epochs = 250

    for epoch in range(epochs):
        running_loss = 0.0
        for data, label in train_loader:
            # ============ Forward ============
            data, label = data.to(device), label.to(device, torch.float)
            output = rob_predictor(data)
            loss = criterion(output, label)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ============ Logging ============
            running_loss += loss.data

        print('Epoch [{}/{}] loss: {:.6f}'.format(epoch + 1, epochs, running_loss))


    # save the model weights
    torch.save(rob_predictor.state_dict(), rob_predictor_weights_PATH)

    print('----------------------------')
    print('Finished training. Model saved.')


# ====================================
# Method: Test the performance of rob predictor, 
# namely computing correlation coefficient and MSE
# ====================================
def test_rob_predictor(classifier, classifier_fmodel, attack, test_loader, epsilons, rob_real_and_predicted_PATH):

    rob_predictor = Rob_predictor().to(device)
    rob_predictor.load_state_dict(torch.load(rob_predictor_weights_PATH))

    # 保存鲁棒性的预测值和真实值
    robust_array = []

    # 计数器
    iterator = 1

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        data_normalized = normalize(data)

        # Forward pass the data through the model
        with torch.no_grad():
            output = classifier(data_normalized)
        # get the index of the max log-probability
        init_pred = output.max(1)[1] 
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        penultimate_layer = get_penultimate_layer(classifier, data_normalized)

        robustness_predict = rob_predictor(torch.tensor(penultimate_layer).to(device))

        # if 0 <= robustness_predict.item() <= 0.031:
        #     robustness_predict_modified = robustness_predict.item()
        # elif robustness_predict.item() < 0:
        #     robustness_predict_modified = 0
        # elif robustness_predict.item() > 0.031:
        #     robustness_predict_modified = 0.031

        # robustness_predict_modified = robustness_predict.item() if robustness_predict.item() >= 0 else 0

        robustness_predict_modified = robustness_predict.item()

        robustness_value = get_robustness_value(classifier_fmodel, attack, data, target, epsilons)

        robust_array.append([robustness_predict_modified, robustness_value])

        print(f"processing img{iterator}, robustness value is {robustness_value:.3f}, roubustness prediction is {robustness_predict_modified:.3f}")
        iterator += 1

        if iterator > 1000:
            break

    # compute the Pearson correlation coefficient
    robust_array = np.array(robust_array)
    pcc = np.corrcoef(robust_array[:, 0], robust_array[:, 1])
    print("pcc is", pcc[0, 1])


    # compute the MSE
    # 均方误差
    mae = sum(np.absolute(robust_array[:, 0] - robust_array[:, 1])) / robust_array.shape[0]
    print("MAE is", mae)

    np.savetxt(rob_real_and_predicted_PATH, robust_array, delimiter=',', header='prediction,truth')




# ================================================================
if __name__ == '__main__':
    use_cuda = True
    device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')
    print("CUDA Available: ", torch.cuda.is_available())


    # load the cifar10 trainset data
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    train_set=datasets.MNIST("./data/datasets/minist/", train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)

    test_set = datasets.MNIST("./data/datasets/minist/", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # ============================================================
    # filenames
    classifier_weights_PATH = "./weights/mnist/mnist_lenet5.pt"
    # This path is to save the training set 
    trainset_for_predictor_PATH = "./data/datasets/rob_predictor/rob_predictor_trainset.pth"
    rob_predictor_weights_PATH = "./weights/rob_predictor/rob_predictor.pt"
    rob_real_and_predicted_PATH = "./data/datasets/rob_predictor/robust_array_resnet32_robust_maxEps.csv"


    print("classifier_weights_PATH:", classifier_weights_PATH)
    print("trainset_for_predictor_PATH:", trainset_for_predictor_PATH)
    print("rob_predictor_weights_PATH:", rob_predictor_weights_PATH)
    print("rob_real_and_predicted_PATH:", rob_real_and_predicted_PATH)

    # Initialize the network

    # classifier = resnet.cifar10_resnet32().to(device)
    classifier=lenet.Lenet5().to(device=device)

    # Load the pretrained model
    classifier.load_state_dict(torch.load(classifier_weights_PATH))
    classifier.eval()

    epsilons = np.arange(.0, .5, .01)

    # normalize =  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize=transforms.Normalize((0.,),(1.,))
    # normalize =  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    # print(normalize)

    # load the adversarial attack
    bounds = (0, 1)
#   mnist 的 mean 和 std = 0.1307 和 0.3081
    # preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
    # preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616], axis=-3)
    # print(preprocessing)
    preprocessing=None
    classifier_fmodel = fb.PyTorchModel(classifier, bounds=bounds, preprocessing=preprocessing)
    attack = fb.attacks.LinfProjectedGradientDescentAttack()


    # ==============================================================================
    t1 = time.time()

    get_dataset_for_rob_predictor(classifier, classifier_fmodel, attack, train_loader, epsilons, trainset_for_predictor_PATH,10000)

    t2 = time.time()

    train_rob_predictor()

    t3 = time.time()

    test_rob_predictor(classifier, classifier_fmodel, attack, test_loader, epsilons, rob_real_and_predicted_PATH)

    t4 = time.time()

    print("classifier_weights_PATH:", classifier_weights_PATH)
    print("get the dataset:", t2-t1)
    print("train rob predictor:", t3-t2)
    print("test rob predictor:", t4-t3)

