# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Torchvision
from torchvision import datasets, transforms



class Lenet1(nn.Module):
    def __init__(self):
        super(Lenet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x


class Lenet4(nn.Module):
    def __init__(self):
        super(Lenet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Lenet5_variant(nn.Module):
    def __init__(self):
        super(Lenet5_variant, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# =================================
def train():

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 10

    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in train_loader:
            # ============ Forward ============
            data, target= data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ============ Logging ============
            running_loss = loss.item()

        print(f'Epoch [{epoch+1}/{epochs}] | loss: {running_loss:.3f}')

        test_log(model)
        
        # torch.save(model.state_dict(), '../weights/image_classifier/mnist_lenet1_{}.pt'.format(epoch+1))
        # print('model_{}.pth saved'.format(epoch+1))

    torch.save(model.state_dict(), model_PATH)
    print('----------------------------')
    print('Finished training. Model saved.')


# =======================================
def test():

    model.load_state_dict(torch.load(model_PATH, map_location=device))
    model.eval()

    correctCnt = 0
    totalCnt = 0
    for data, target in test_loader:
            data, target= data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            init_pred = output.max(1)[1]
            correctCnt += (init_pred == target).sum().item()
            totalCnt += data.shape[0]
            if totalCnt % 512 == 0:
                print(totalCnt, "samples finished.")

    print('----------------------------')
    print('test accuracy is ', correctCnt/totalCnt)


# ==================================
def test_log(model):

    model.eval()

    correctCnt = 0
    totalCnt = 0
    with torch.no_grad():
        for data_test, target_test in test_loader:
            data_test, target_test= data_test.to(device), target_test.to(device)
            output = model(data_test)
            init_pred = output.max(1)[1]
            correctCnt += (init_pred == target_test).sum().item()
            totalCnt += data_test.shape[0]
    print('test accuracy is ', correctCnt/totalCnt)



# =====================================================
if __name__ == '__main__':
    use_cuda = True
    device = 'cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu'
    print("CUDA Available: ", torch.cuda.is_available())

    # Load data
    transform = transforms.Compose([transforms.ToTensor(), ])

    train_set = datasets.MNIST('E:\code\DataSets', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)

    test_set = datasets.MNIST('E:\code\DataSets', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=True)



    model_PATH = '../weights/image_classifier/mnist_lenet5.pt'

    model = Lenet5().to(device)

    # train()

    test()