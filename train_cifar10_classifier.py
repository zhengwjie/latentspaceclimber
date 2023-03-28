from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pprint import pprint
from torchvision.datasets import CIFAR10
from torchvision import transforms

from torch.nn import functional as F
from logutil import TimeSeries


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--cuda', default=True, action="store_true", help='Use CUDA (requires GPU)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

opt = parser.parse_args()
print('Running with options:')
pprint(opt)

def get_cifar(training=False):
    # CIFAR stands for Canadian Institute for Advanced Research
    # The CIFAR dataset consists of 32x32 RGB images of 10 categories
    dataset = CIFAR10(root='..', download=True, train=training,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)


train_dataloader = get_cifar(training=True)
test_dataloader = get_cifar(training=False)

device = torch.device("cuda" if opt.cuda else "cpu")


class Classifier(nn.Module):
    def __init__(self):
        # This is a fully-connected neural network ("multilayer perceptron")
        # For better results, try adding convolutional layers
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 256)
        self.drop1 = nn.Dropout(.2)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(.2)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x, ts):
        # The input image is in standard "BCHW" format
        # To use an MLP, we reshape it to a 1D vector
        batch_size, channels, height, width = x.shape
        x = x.reshape((batch_size, -1))

        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.bn1(x)
        x = self.drop1(x)
        ts.collect('Layer 1 Activation Mean', x.mean())
        ts.collect('Layer 1 Activation Variance', x.var(0).mean())

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.bn2(x)
        x = self.drop2(x)
        ts.collect('Layer 2 Activation Mean', x.mean())
        ts.collect('Layer 2 Activation Variance', x.var(0).mean())

        x = self.fc3(x)
        ts.collect('Layer 3 Activation Mean', x.mean())
        ts.collect('Layer 3 Activation Variance', x.var(0).mean())
        x = F.softmax(x, dim=1)
        return x


netC = Classifier().to(device)
optimizerC = optim.Adam(netC.parameters(), lr=opt.lr)

total_batches = len(train_dataloader) + len(test_dataloader)
ts = TimeSeries('CIFAR10', opt.epochs * total_batches)

for epoch in range(opt.epochs):
    for data_batch, labels in train_dataloader:
        # t.to(device) is equivalent to t.cuda()
        data_batch = data_batch.to(device)
        labels = labels.to(device)

        # At the center of a Pytorch training loop are three operations:
        # parameters.zero_grad() to reset accumulated gradients
        # loss.backward() to accumulate a gradient for some loss function
        # optimizer.step() to update parameters using the gradient
        netC.zero_grad()
        predictions = netC(data_batch, ts)
        loss = F.cross_entropy(predictions, labels)
        loss.backward()
        optimizerC.step()

        # For classification, we want to track accuracy during training
        # Note that accuracy is our true objective, but we optimize with
        # cross_entropy because it is smoothly differentiable
        pred_confidence, pred_argmax = predictions.max(dim=1)
        correct = torch.sum(pred_argmax == labels)
        accuracy = float(correct) / len(data_batch)

        ts.collect('Training Loss', loss)
        ts.collect('Training Accuracy', accuracy)
        ts.print_every(n_sec=4)

    total_correct = 0
    for data_batch, labels in test_dataloader:
        data_batch = data_batch.to(device)
        labels = labels.to(device)

        predictions = netC(data_batch, ts)
        pred_confidence, pred_argmax = predictions.max(dim=1)
        correct = torch.sum(pred_argmax == labels)
        accuracy = float(correct) / len(data_batch)
        total_correct += correct

        ts.collect('Testing Loss', loss)
        ts.collect('Testing Accuracy', accuracy)
        ts.print_every(n_sec=4)
print(ts)
print('Final results: {} correct out of {}'.format(total_correct, len(test_dataloader.dataset)))