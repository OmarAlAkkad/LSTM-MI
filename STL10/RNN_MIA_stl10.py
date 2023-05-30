# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:00:26 2023

@author: omars
"""
#@title (Run) Part 1: Define required functions for Data processing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import Tensor
import os
import argparse
import math
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix, accuracy_score
import pandas as pd
import random
import numpy as np
#@title (Run) Part 3: Prepare stl1010 dataset for target and shadow model
import pickle

#@title (Run) Part 4.2: Define required functions for DLA & DLA+RNN
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

def balance_val_split(dataset, train_size=12500):

    try:
        targets = np.array(dataset.targets)
    except:
        targets = []  # create an empty list to store the targets
        for data in dataset.datasets:
            targets += data.targets  # concatenate the targets from each dataset into the list
        targets = np.array(targets)
    #targets = np.array(dataset.datasets.targets)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        train_size=train_size,
        stratify=targets
    )
    train_dataset = Subset(dataset, indices=train_indices)
    # Get the data from the subset dataset
    subset_data = [train_dataset[idx][0] for idx in range(len(train_dataset))]
    subset_labels = [train_dataset[idx][1] for idx in range(len(train_dataset))]
    # Create a dataset from the list of data and targets
    train_dataset = MyDataset(subset_data, subset_labels)


    val_dataset = Subset(dataset, indices=val_indices)
    # Get the data from the subset dataset
    subset_data = [val_dataset[idx][0] for idx in range(len(val_dataset))]
    subset_labels = [val_dataset[idx][1] for idx in range(len(val_dataset))]
    # Create a dataset from the list of data and targets
    val_dataset = MyDataset(subset_data, subset_labels)

    return train_dataset, val_dataset


def count_label_frequency(target_train_dataset):
	from collections import Counter
	target_labels = []  # create an empty list to store the labels

	for i in range(len(target_train_dataset)):
			_, label = target_train_dataset[i]  # extract the label for the i-th example in the subset
			target_labels.append(label)  # append the label to the 'subset_labels' list


	return Counter(target_labels)



def custom_transform(image: Tensor) -> Tensor:
    import random
    # randomly flip horizontally or vertically with 25% chance
    if random.random() < 0.25:
        image = RandomHorizontalFlip(p=1)(image)
    elif random.random() < 0.5:
        image = RandomVerticalFlip(p=1)(image)

    # randomly shift the image by 2 pixels to the left or right with 25% chance
    if random.random() < 0.25:
        image = RandomCrop((image.shape[-2], image.shape[-1] - 2), pad_if_needed=False)(image)
    elif random.random() < 0.5:
        image = RandomCrop((image.shape[-2], image.shape[-1] + 2), pad_if_needed=False)(image)

    # randomly shift the image by 2 pixels to the top or bottom with 25% chance
    if random.random() < 0.25:
        image = RandomCrop((image.shape[-2] - 2, image.shape[-1]), pad_if_needed=False)(image)
    elif random.random() < 0.5:
        image = RandomCrop((image.shape[-2] + 2, image.shape[-1]), pad_if_needed=False)(image)

    return image

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck1(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, enable_RNN='LSTM'):
        super(DenseNet, self).__init__()



        self.enable_RNN = enable_RNN
        if enable_RNN not in ['None', 'LSTM', 'Bi-LSTM']:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")

        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        #self.linear = nn.Linear(num_planes, num_classes)


        if enable_RNN == 'None':
          self.linear = nn.Linear(num_planes, num_classes)
        elif enable_RNN == 'LSTM':
          self.rnn = nn.LSTM(num_planes, 1024, dropout = 0.6)
          self.linear = nn.Linear(1024, num_classes)
        elif enable_RNN == 'Bi-LSTM':
          self.rnn = nn.LSTM(num_planes, 2048, 1, dropout = 0.3, batch_first=True, bidirectional=True)
          self.linear = nn.Linear(4096, num_classes)
        else:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")




    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        #out = out.view(out.size(0), -1)
        #out = self.linear(out)

        if self.enable_RNN == 'None':
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        else:
          # add LSTM
          #print(out.shape)
          out = out.view(out.size(0), 1,  -1)
          out,_ = self.rnn(out)
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck1, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck1, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck1, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck1, [6,12,36,24], growth_rate=48)

def densenet_stl10():
    return DenseNet(Bottleneck1, [6,12,24,16], growth_rate=12)



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, enable_RNN='LSTM'):
        super(ResNet, self).__init__()


        self.enable_RNN = enable_RNN
        if enable_RNN not in ['None', 'LSTM', 'Bi-LSTM']:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.linear = nn.Linear(512*block.expansion, num_classes)



        if enable_RNN == 'None':
          self.linear = nn.Linear(512*block.expansion, num_classes)
        elif enable_RNN == 'LSTM':
          self.rnn = nn.LSTM(512*block.expansion, 1024, dropout = 0.6)
          self.linear = nn.Linear(1024, num_classes)
        elif enable_RNN == 'Bi-LSTM':
          self.rnn = nn.LSTM(512*block.expansion, 2048, 1, dropout = 0.3, batch_first=True, bidirectional=True)
          self.linear = nn.Linear(4096, num_classes)
        else:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        if self.enable_RNN == 'None':
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        else:
          # add LSTM
          #print(out.shape)
          out = out.view(out.size(0), 1,  -1)
          out,_ = self.rnn(out)
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class DLA(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10, enable_RNN='LSTM'):
        super(DLA, self).__init__()

        self.enable_RNN = enable_RNN
        if enable_RNN not in ['None', 'LSTM', 'Bi-LSTM']:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")

        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(4096, num_classes)

        if enable_RNN == 'None':
          self.linear = nn.Linear(512, num_classes)
        elif enable_RNN == 'LSTM':
          self.rnn = nn.LSTM(512, 1024, dropout = 0.6)
          self.linear = nn.Linear(1024, num_classes)
        elif enable_RNN == 'Bi-LSTM':
          self.rnn = nn.LSTM(512, 2048, 1, dropout = 0.3, batch_first=True, bidirectional=True)
          self.linear = nn.Linear(4096, num_classes)
        else:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)

        if self.enable_RNN == 'None':
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        else:
          # add LSTM
          #print(out.shape)
          out = out.view(out.size(0), 1,  -1)
          out,_ = self.rnn(out)
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        return out

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, enable_RNN='LSTM'):
        super(VGG, self).__init__()


        self.enable_RNN = enable_RNN
        if enable_RNN not in ['None', 'LSTM', 'Bi-LSTM']:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")


        self.features = self._make_layers(cfg[vgg_name])


        if enable_RNN == 'None':
          self.linear = nn.Linear(512, num_classes)
        elif enable_RNN == 'LSTM':
          self.rnn = nn.LSTM(512, 1024, dropout = 0.6)
          self.linear = nn.Linear(1024, num_classes)
        elif enable_RNN == 'Bi-LSTM':
          self.rnn = nn.LSTM(512, 2048, 1, dropout = 0.3, batch_first=True, bidirectional=True)
          self.linear = nn.Linear(4096, num_classes)
        else:
          raise Exception("enable_RNN only supports one of ['None', 'LSTM', 'Bi-LSTM']")


    def forward(self, x):
        out = self.features(x)

        if self.enable_RNN == 'None':
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        else:
          # add LSTM
          #print(out.shape)
          out = out.view(out.size(0), 1,  -1)
          out,_ = self.rnn(out)
          out = out.view(out.size(0), -1)
          out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


#@title (Run) Part 2: Define required functions for Data Training

# Training
def train(trainloader, epoch, batch_size=128, logfile = "train.summary"):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        if inputs.shape[0] != batch_size:
          print(inputs.shape)
          continue
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 30 == 0:
                print(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(len(trainloader), 'Epoch: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    f = open(logfile, "a")
    f.write('Epoch: %d | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)\n'
                     % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    f.close()

def test(testloader, epoch, batch_size=128, logfile = "train.summary", save_modelpath = './DLA'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0



    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 30 == 0:
                print(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(len(testloader), 'Epoch: %d | Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    f = open(logfile, "a")
    f.write('Epoch: %d | Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)\n'
                         % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    f.close()
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_modelpath):
            os.mkdir(save_modelpath)
        torch.save(state, save_modelpath+'/ckpt.pth')
        best_acc = acc

def draw_training_summary(filepath = 'target_train_DCA-BiLSTM.summary'):
    import matplotlib.pyplot as plt
    import numpy as np

    with open(filepath, 'r') as f:
        results_summary = f.read()

    train_epoch = []
    train_loss = []
    train_acc = []
    test_epoch = []
    test_loss=[]
    test_acc=[]
    for line in results_summary.split("\n"):
        try:
            r_epoch = line.split('|')[0].strip().split(' ')[1]
            r_loss = line.split('|')[1].strip().split(' ')[2].replace('%','')
            r_acc = line.split('|')[2].strip().split(' ')[2].replace('%','')
            if 'Train' in line:
                train_epoch.append(int(r_epoch))
                train_loss.append(float(r_loss))
                train_acc.append(float(r_acc))
            if 'Test' in line:
                test_epoch.append(int(r_epoch))
                test_loss.append(float(r_loss))
                test_acc.append(float(r_acc))
        except:
            print(line)

    # Create a new figure and plot the data
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(train_acc, label='Train')
    plt.plot(test_acc, label='Test')
    plt.axhline(y=np.max(test_acc), color='r', linestyle='--')
    # Add text for the horizontal line
    plt.text(test_epoch[-10], np.max(test_acc)*1.05, np.max(test_acc), color='r', fontsize=10)
    # Customize the plot
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')

    # Customize the plot
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Display the plot
    plt.show()
#@title (Run) Part 3: Prepare stl1010 dataset for target and shadow model

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def create_stl10_dataset_torch(name, load_data = True, batch_size=128, target_train_size = 15000, target_test_size= 15000, shadow_train_size = 15000, shadow_test_size= 15000):

  # Data
  print('==> Preparing data..')
  if load_data:

    try:
          data_file = open(f'target_trainloader_{name}.p', 'rb')
          target_trainloader = pickle.load(data_file)
          data_file.close()
          data_file = open(f'target_testloader_{name}.p', 'rb')
          target_testloader = pickle.load(data_file)
          data_file.close()
          data_file = open(f'shadow_trainloader_{name}.p', 'rb')
          shadow_trainloader = pickle.load(data_file)
          data_file.close()
          data_file = open(f'shadow_test_dataset_{name}.p', 'rb')
          shadow_testloader = pickle.load(data_file)
          data_file.close()

    except:

          transform = transforms.Compose([
              transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])

          stl10_trainset = torchvision.datasets.STL10(
              root='./data', split = 'train', download=True, transform=transform)


          stl10_testset = torchvision.datasets.STL10 (
              root='./data', split = 'test', download=True, transform=transform)

          stl10_dataset = torch.utils.data.ConcatDataset([stl10_trainset, stl10_testset])


          #target_train_size = int(0.25 * len(stl10_dataset)) # 15000
          remain_size = len(stl10_dataset) - target_train_size
          target_train_dataset, remain_dataset = torch.utils.data.random_split(stl10_dataset, [target_train_size, remain_size])

          #target_test_size = int(0.25 * len(stl10_dataset)) # 15000
          remain_size = len(remain_dataset) - target_test_size
          target_test_dataset, remain_dataset = torch.utils.data.random_split(remain_dataset, [target_test_size, remain_size])

          #target_test_dataset, remain_dataset = balance_val_split(remain_dataset, train_size=target_test_size)


          #shadow_train_size = int(0.25 * len(stl10_dataset)) # 15000
          remain_size = len(remain_dataset) - shadow_train_size
          shadow_train_dataset, shadow_test_dataset = torch.utils.data.random_split(remain_dataset, [shadow_train_size, remain_size])
          #shadow_train_dataset, shadow_test_dataset = balance_val_split(remain_dataset, train_size=shadow_train_size)

          print("Setting target_train_dataset size to ",len(target_train_dataset), count_label_frequency(target_train_dataset))
          print("Setting target_test_dataset size to ",len(target_test_dataset), count_label_frequency(target_test_dataset))
          print("Setting shadow_train_dataset size to ",len(shadow_train_dataset), count_label_frequency(shadow_train_dataset))
          print("Setting shadow_test_dataset size to ",len(shadow_test_dataset), count_label_frequency(shadow_test_dataset))
          #print("Setting testset size to ",len(testset))



          '''
          transform_train = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])
          '''



          transform_train = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              custom_transform,
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

          transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

          # apply the data augmentation transformations to the subset
          target_train_dataset.dataset.transform = transform_train
          # Load the transformed subset using a DataLoader
          target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)


          target_test_dataset.dataset.transform = transform_test
          # Load the transformed subset using a DataLoader
          target_testloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=True, drop_last = True)


          # apply the data augmentation transformations to the subset
          shadow_train_dataset.dataset.transform = transform_train
          # Load the transformed subset using a DataLoader
          shadow_trainloader = DataLoader(shadow_train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)


          shadow_test_dataset.dataset.transform = transform_test
          # Load the transformed subset using a DataLoader
          shadow_testloader = DataLoader(shadow_test_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

          pickle.dump(target_trainloader, open(f'target_trainloader_{name}.p', 'wb'))
          pickle.dump(target_testloader, open(f'target_testloader_{name}.p', 'wb'))
          pickle.dump(shadow_trainloader, open(f'shadow_trainloader_{name}.p', 'wb'))
          pickle.dump(shadow_testloader, open(f'shadow_test_dataset_{name}.p', 'wb'))

    else:
          transform = transforms.Compose([
              transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])

          stl10_trainset = torchvision.datasets.STL10(
              root='./data', split='train', download=True, transform=transform)


          stl10_testset = torchvision.datasets.STL10 (
              root='./data', split='test', download=True, transform=transform)

          stl10_dataset = torch.utils.data.ConcatDataset([stl10_trainset, stl10_testset])


          #target_train_size = int(0.25 * len(stl10_dataset)) # 15000
          remain_size = len(stl10_dataset) - target_train_size
          target_train_dataset, remain_dataset = torch.utils.data.random_split(stl10_dataset, [target_train_size, remain_size])

          #target_test_size = int(0.25 * len(stl10_dataset)) # 15000
          remain_size = len(remain_dataset) - target_test_size
          target_test_dataset, remain_dataset = torch.utils.data.random_split(remain_dataset, [target_test_size, remain_size])

          #target_test_dataset, remain_dataset = balance_val_split(remain_dataset, train_size=target_test_size)


          #shadow_train_size = int(0.25 * len(stl10_dataset)) # 15000
          remain_size = len(remain_dataset) - shadow_train_size
          shadow_train_dataset, shadow_test_dataset = torch.utils.data.random_split(remain_dataset, [shadow_train_size, remain_size])
          #shadow_train_dataset, shadow_test_dataset = balance_val_split(remain_dataset, train_size=shadow_train_size)

          print("Setting target_train_dataset size to ",len(target_train_dataset), count_label_frequency(target_train_dataset))
          print("Setting target_test_dataset size to ",len(target_test_dataset), count_label_frequency(target_test_dataset))
          print("Setting shadow_train_dataset size to ",len(shadow_train_dataset), count_label_frequency(shadow_train_dataset))
          print("Setting shadow_test_dataset size to ",len(shadow_test_dataset), count_label_frequency(shadow_test_dataset))
          #print("Setting testset size to ",len(testset))



          '''
          transform_train = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])
          '''



          transform_train = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              custom_transform,
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

          transform_test = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
          ])

          # apply the data augmentation transformations to the subset
          target_train_dataset.dataset.transform = transform_train
          # Load the transformed subset using a DataLoader
          target_trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)


          target_test_dataset.dataset.transform = transform_test
          # Load the transformed subset using a DataLoader
          target_testloader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=True, drop_last = True)


          # apply the data augmentation transformations to the subset
          shadow_train_dataset.dataset.transform = transform_train
          # Load the transformed subset using a DataLoader
          shadow_trainloader = DataLoader(shadow_train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)


          shadow_test_dataset.dataset.transform = transform_test
          # Load the transformed subset using a DataLoader
          shadow_testloader = DataLoader(shadow_test_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

          pickle.dump(target_trainloader, open(f'target_trainloader_{name}.p', 'wb'))
          pickle.dump(target_testloader, open(f'target_testloader_{name}.p', 'wb'))
          pickle.dump(shadow_trainloader, open(f'shadow_trainloader_{name}.p', 'wb'))
          pickle.dump(shadow_testloader, open(f'shadow_test_dataset_{name}.p', 'wb'))


  return target_trainloader, target_testloader, shadow_trainloader, shadow_testloader

if __name__ == "__main__":
    models = [('DLA','DLA-BiLSTM','./Target-DLA-BiLSTM_models/','DLA-BiLSTM-Target'),('DLA','DLA-BiLSTM','./Shadow-DLA-BiLSTM_models/','DLA-BiLSTM-Shadow'),
              ('DLA','DLA-LSTM','./Target-DLA-LSTM_models/','DLA-LSTM-Target'),('DLA','DLA-LSTM','./Shadow-DLA-LSTM_models/','DLA-LSTM-Shadow'),
              ('DLA','DLA','./Target-DLA_models/','DLA-Target'),('DLA','DLA','./Shadow-DLA_models/','DLA-Shadow'),
              ('resnet','ResNet18-BiLSTM','./Target-ResNet18-BiLSTM_models/','ResNet18-BiLSTM-Target'),('resnet','ResNet18-BiLSTM','./Shadow-ResNet18-BiLSTM_models/','ResNet18-BiLSTM-Shadow'),
              ('resnet','ResNet18-LSTM','./Target-ResNet18-LSTM_models/','ResNet18-LSTM-Target'),('resnet','ResNet18-LSTM','./Shadow-ResNet18-LSTM_models/','ResNet18-LSTM-Shadow'),
              ('resnet','ResNet18','./Target-ResNet18_models/','ResNet18-Target'),('resnet','ResNet18','./Shadow-ResNet18_models/','ResNet18-Shadow'),
              ('densenet','DenseNet121-BiLSTM','./Target-DenseNet121-BiLSTM_models/','DenseNet121-BiLSTM-Target'),('densenet','DenseNet121-BiLSTM','./Shadow-DenseNet121-BiLSTM_models/','DenseNet121-BiLSTM-Shadow'),
              ('densenet','DenseNet121-LSTM','./Target-DenseNet121-LSTM_models/','DenseNet121-LSTM-Target'),('densenet','DenseNet121-LSTM','./Shadow-DenseNet121-LSTM_models/','DenseNet121-LSTM-Shadow'),
              ('densenet','DenseNet121','./Target-DenseNet121_models/','DenseNet121-Target'),('densenet','DenseNet121','./Shadow-DenseNet121_models/','DenseNet121-Shadow'),
              ('VGG','VGG-BiLSTM','./Target-VGG-BiLSTM_models/','VGG-BiLSTM-Target'),('VGG','VGG-BiLSTM','./Shadow-VGG-BiLSTM_models/','VGG-BiLSTM-Shadow'),
              ('VGG','VGG-LSTM','./Target-VGG-LSTM_models/','VGG-LSTM-Target'),('VGG','VGG-LSTM','./Shadow-VGG-LSTM_models/','VGG-LSTM-Shadow'),
              ('VGG','VGG','./Target-VGG_models/','VGG-Target'),('VGG','VGG','./Shadow-VGG_models/','VGG-Shadow')]

    models = [
              ('densenet','DenseNet121-BiLSTM','./Target-DenseNet121-BiLSTM_models/','DenseNet121-BiLSTM-Target'),('densenet','DenseNet121-BiLSTM','./Shadow-DenseNet121-BiLSTM_models/','DenseNet121-BiLSTM-Shadow'),
              ('densenet','DenseNet121-LSTM','./Target-DenseNet121-LSTM_models/','DenseNet121-LSTM-Target'),('densenet','DenseNet121-LSTM','./Shadow-DenseNet121-LSTM_models/','DenseNet121-LSTM-Shadow'),
              ('densenet','DenseNet121','./Target-DenseNet121_models/','DenseNet121-Target'),('densenet','DenseNet121','./Shadow-DenseNet121_models/','DenseNet121-Shadow'),
              ('VGG','VGG-BiLSTM','./Target-VGG-BiLSTM_models/','VGG-BiLSTM-Target'),('VGG','VGG-BiLSTM','./Shadow-VGG-BiLSTM_models/','VGG-BiLSTM-Shadow'),
              ('VGG','VGG-LSTM','./Target-VGG-LSTM_models/','VGG-LSTM-Target'),('VGG','VGG-LSTM','./Shadow-VGG-LSTM_models/','VGG-LSTM-Shadow'),
              ('VGG','VGG','./Target-VGG_models/','VGG-Target'),('VGG','VGG','./Shadow-VGG_models/','VGG-Shadow')]

    for data,method_name,save_model_folder,name in models:

        target_trainloader, target_testloader, shadow_trainloader, shadow_testloader = create_stl10_dataset_torch(data, load_data = True, batch_size=64, target_train_size = 3250, target_test_size= 3250, shadow_train_size = 3250, shadow_test_size= 3250)
        batch_size = 64  #@param {type:"integer"}
        load_pretrain_weight = False   #@param {type:"boolean"}
        print('==> Building model for ' + method_name)
        if method_name == 'DLA-BiLSTM':
          # Model
          net = DLA(num_classes=10, enable_RNN='Bi-LSTM')
          net.cuda()
        elif method_name == 'DLA-LSTM':
          # Model
          net = DLA(num_classes=10, enable_RNN='LSTM')
          net.cuda()
        elif method_name == 'DLA':
          # Model
          net = DLA(num_classes=10, enable_RNN='None')
          net.cuda()
        elif method_name == 'ResNet18-BiLSTM':
            # Model
          net = ResNet(BasicBlock, [2, 2, 2, 2], enable_RNN='Bi-LSTM')
          net.cuda()
        elif method_name == 'ResNet18-LSTM':
          # Model
          net = ResNet(BasicBlock, [2, 2, 2, 2], enable_RNN='LSTM')
          net.cuda()
        elif method_name == 'ResNet18':
          # Model
          net = ResNet(BasicBlock, [2, 2, 2, 2], enable_RNN='None')
          net.cuda()
        elif method_name == 'VGG-BiLSTM':
      # Model
          net = VGG('VGG11', enable_RNN='Bi-LSTM')
          net.cuda()
        elif method_name == 'VGG-LSTM':
          # Model
          net = VGG('VGG11', enable_RNN='LSTM')
          net.cuda()
        elif method_name == 'VGG':
          # Model
          net = VGG('VGG11', enable_RNN='None')
          net.cuda()
        elif method_name == 'DenseNet121-BiLSTM':
      # Model
          net = DenseNet(Bottleneck1, [6,12,24,16], growth_rate=32, enable_RNN='Bi-LSTM')
          net.cuda()
        elif method_name == 'DenseNet121-LSTM':
          # Model
          net = DenseNet(Bottleneck1, [6,12,24,16], growth_rate=32, enable_RNN='LSTM')
          net.cuda()
        elif method_name == 'DenseNet121':
          # Model
          net = DenseNet(Bottleneck1, [6,12,24,16], growth_rate=32, enable_RNN='None')
          net.cuda()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if load_pretrain_weight:
          try:
              # Load checkpoint.
              print('==> Resuming from checkpoint..')
              checkpoint = torch.load(save_model_folder+'/ckpt.pth')
              net.load_state_dict(checkpoint['net'])
              best_acc = checkpoint['acc']
              start_epoch = checkpoint['epoch']
          except:
              print('!!! Error: no checkpoint directory found!')
        else:
          print('==> Training model from scratch..')

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total trained parameters: ",pytorch_total_params)

        max_epoch = 30  #@param {type:"integer"}
        train_result_summary = f'{name}.summary'   #@param {type:"string"}

        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        if not load_pretrain_weight:
            f = open(train_result_summary, "w")
            f.write('')
            f.close()

        if name[-1] == 't':
            for epoch in range(start_epoch, start_epoch+max_epoch):
                train(target_trainloader, epoch, batch_size=batch_size, logfile = train_result_summary)
                test(target_testloader, epoch, batch_size=batch_size, logfile = train_result_summary, save_modelpath = save_model_folder)

        elif name[-1] == 'w':
            for epoch in range(start_epoch, start_epoch+max_epoch):
                train(shadow_trainloader, epoch, batch_size=batch_size, logfile = train_result_summary)
                test(shadow_testloader, epoch, batch_size=batch_size, logfile = train_result_summary, save_modelpath = save_model_folder)



