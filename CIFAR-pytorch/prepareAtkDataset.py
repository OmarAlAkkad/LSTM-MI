# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 00:05:11 2023

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
#@title (Run) Part 3: Prepare Cifar10 dataset for target and shadow model
import pickle
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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
          out1 = out.view(out.size(0), -1)
          out = self.linear(out1)
        return out,out1

def DenseNet121():
    return DenseNet(Bottleneck1, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck1, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck1, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck1, [6,12,36,24], growth_rate=48)

def densenet_cifar():
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
          out1 = out.view(out.size(0), -1)
          out = self.linear(out1)
        return out,out1


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
          out1 = out.view(out.size(0), -1)
          out = self.linear(out1)
        return out,out1

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
          out1 = out.view(out.size(0), -1)
          out = self.linear(out1)
        return out,out1

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


def load_data(model):
  print(f"Loading Data for {model}")
  data_file = open(f'target_trainloader_{model}.p', 'rb')
  target_trainloader = pickle.load(data_file)
  data_file.close()
  data_file = open(f'target_testloader_{model}.p', 'rb')
  target_testloader = pickle.load(data_file)
  data_file.close()
  data_file = open(f'shadow_trainloader_{model}.p', 'rb')
  shadow_trainloader = pickle.load(data_file)
  data_file.close()
  data_file = open(f'shadow_test_dataset_{model}.p', 'rb')
  shadow_testloader = pickle.load(data_file)
  data_file.close()

  return target_trainloader, target_testloader, shadow_trainloader, shadow_testloader

def get_attack_features(dataset, lstm = True):
    print("Getting Attack Features")
    predictions = []
    labels = []
    losses = []
    lstm_list = []
    softmax = torch.nn.Softmax(dim=1)
    if lstm:
        for batch_idx, (inputs, targets) in enumerate(dataset):
                #inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs, lstm_neurons = net(inputs)
                for i in range(len(outputs)):
                    l = criterion(outputs[i].view(1,10),targets[i].view(1))
                    losses.append(l.item())
                outputs = softmax(outputs)
                predictions.extend(outputs.tolist())
                lstm_list.extend(lstm_neurons.tolist())
                labels.extend(targets.tolist())
                predictions_labels = []
        for pred in predictions:
            predictions_labels.append(np.argmax(pred, axis=0))

        return predictions,labels,losses, predictions_labels,lstm_list

    else:
        for batch_idx, (inputs, targets) in enumerate(dataset):
                #inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs,_ = net(inputs)
                for i in range(len(outputs)):
                    l = criterion(outputs[i].view(1,10),targets[i].view(1))
                    losses.append(l.item())
                outputs = softmax(outputs)
                predictions.extend(outputs.tolist())
                labels.extend(targets.tolist())

        predictions_labels = []
        for pred in predictions:
            predictions_labels.append(np.argmax(pred, axis=0))

        return predictions,labels,losses, predictions_labels

def prepare_dataframe(inputs,all_labels,predictions,labels,losses,label = 1,include_losses = True, include_labels = True):
    print("Preparing data frame")
    for i in range(len(predictions)):
        if include_losses:
            predictions[i].append(losses[i])
        if include_labels:
            predictions[i].append(labels[i])
        inputs.append(predictions[i])
        all_labels.append(label)

    return inputs, all_labels

def prepare_lstm_dataframe(inputs,all_labels,predictions,labels,losses,lstm,label = 1,include_losses = True, include_labels = True, include_lstm = True):
    print("Preparing LSTM data frame")
    for i in range(len(predictions)):
        if include_losses:
            predictions[i].append(losses[i])
        if include_labels:
            predictions[i].append(labels[i])
        if include_lstm:
            predictions[i].extend(lstm[i])
        inputs.append(predictions[i])
        all_labels.append(label)

    return inputs, all_labels

def create_dataframe(name, inputs, all_labels):
    print(f"creating data frame for {name}")
    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs,
        'Labels': all_labels
        }
    dataframe = pd.DataFrame(data=d)

    pickle.dump(dataframe, open(f'{name}_dataframe.p', 'wb'))

    return dataframe

def create_statistics_dataframe(name,train_predictions_labels, y_train_label, test_predictions_labels, y_test_label):
    print(f"creating statistics dataframe {name}")
    sets = ["train", "test"]
    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []
    for set_type in sets:
        locals()[f'{set_type}_accuracy'] = accuracy_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
        locals()[f'confusion_matrix_{set_type}'] = confusion_matrix(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
        locals()[f'TN_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][0][0]
        locals()[f'FP_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][0][1]
        locals()[f'FN_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][1][0]
        locals()[f'TP_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][1][1]
        locals()[f'Negative_recall_{set_type}'] = locals()[f'TN_{set_type}'] / (locals()[f'TN_{set_type}'] + locals()[f'FP_{set_type}'])
        locals()[f'precision_{set_type}'] = precision_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='macro')
        locals()[f'recall_{set_type}'] = recall_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='macro')

        locals()[f'f1_{set_type}'] = f1_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'], average='macro')
        models.append(f'model')
        Data.append(set_type)
        Accuracy.append(locals()[f'{set_type}_accuracy'])
        Precision.append(locals()[f'precision_{set_type}'])
        Recall.append(locals()[f'recall_{set_type}'])
        Negative_Recall.append( locals()[f'Negative_recall_{set_type}'])
        F1_Score.append(locals()[f'f1_{set_type}'])

    d = pd.DataFrame({'Model' : models,
         'Data': Data,
         'Accuracy': Accuracy,
         'Precision': Precision,
         'Recall': Recall,
         'Negative Recall': Negative_Recall,
         'F1 Score': F1_Score,
         })

    d.to_csv(f'{name}.csv')

    return d


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

    LSTM_models =[('DLA','DLA-BiLSTM','./Target-DLA-BiLSTM_models/','DLA-BiLSTM-Target'),('DLA','DLA-BiLSTM','./Shadow-DLA-BiLSTM_models/','DLA-BiLSTM-Shadow'),
               ('DLA','DLA-LSTM','./Target-DLA-LSTM_models/','DLA-LSTM-Target'),('DLA','DLA-LSTM','./Shadow-DLA-LSTM_models/','DLA-LSTM-Shadow'),
               ('resnet','ResNet18-BiLSTM','./Target-ResNet18-BiLSTM_models/','ResNet18-BiLSTM-Target'),('resnet','ResNet18-BiLSTM','./Shadow-ResNet18-BiLSTM_models/','ResNet18-BiLSTM-Shadow'),
               ('resnet','ResNet18-LSTM','./Target-ResNet18-LSTM_models/','ResNet18-LSTM-Target'),('resnet','ResNet18-LSTM','./Shadow-ResNet18-LSTM_models/','ResNet18-LSTM-Shadow'),
               ('densenet','DenseNet121-BiLSTM','./Target-DenseNet121-BiLSTM_models/','DenseNet121-BiLSTM-Target'),('densenet','DenseNet121-BiLSTM','./Shadow-DenseNet121-BiLSTM_models/','DenseNet121-BiLSTM-Shadow'),
               ('densenet','DenseNet121-LSTM','./Target-DenseNet121-LSTM_models/','DenseNet121-LSTM-Target'),('densenet','DenseNet121-LSTM','./Shadow-DenseNet121-LSTM_models/','DenseNet121-LSTM-Shadow'),
               ('VGG','VGG-BiLSTM','./Target-VGG-BiLSTM_models/','VGG-BiLSTM-Target'),('VGG','VGG-BiLSTM','./Shadow-VGG-BiLSTM_models/','VGG-BiLSTM-Shadow'),
               ('VGG','VGG-LSTM','./Target-VGG-LSTM_models/','VGG-LSTM-Target'),('VGG','VGG-LSTM','./Shadow-VGG-LSTM_models/','VGG-LSTM-Shadow')]
    lstm = True
    for data,method_name,save_model_folder,name in LSTM_models:
        target_trainloader, target_testloader, shadow_trainloader, shadow_testloader = load_data(data)
        batch_size = 64  #@param {type:"integer"}
        load_pretrain_weight = True   #@param {type:"boolean"}

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

        if name[-1] == 't':
            if lstm:
                target_train_predictions,target_train_labels,target_train_losses,target_train_prediction_labels,target_train_lstm = get_attack_features(target_trainloader,lstm)
                target_test_predictions,target_test_labels,target_test_losses,target_test_prediction_labels,target_test_lstm = get_attack_features(target_testloader,lstm)
                target_inputs, target_labels = [], []
                target_inputs, target_labels = prepare_lstm_dataframe(target_inputs, target_labels,target_train_predictions,target_train_labels,target_train_losses,target_train_lstm, label = 1,include_losses = True, include_labels = True,include_lstm = True)
                target_inputs, target_labels = prepare_lstm_dataframe(target_inputs, target_labels,target_test_predictions,target_test_labels,target_test_losses,target_train_lstm, label = 0,include_losses = True, include_labels = True,include_lstm = True)
            else:
                target_dataframe = create_dataframe(name, target_inputs, target_labels)
                target_d = create_statistics_dataframe(name,target_train_prediction_labels, target_train_labels, target_test_prediction_labels, target_test_labels)
        elif name[-1] == 'w':
            if lstm:
                shadow_train_predictions,shadow_train_labels,shadow_train_losses,shadow_train_prediction_labels, shadow_train_lstm = get_attack_features(shadow_trainloader,lstm)
                shadow_test_predictions,shadow_test_labels,shadow_test_losses,shadow_test_prediction_labels, shadow_test_lstm = get_attack_features(shadow_testloader,lstm)
                shadow_inputs, shadow_labels = [], []
                shadow_inputs, shadow_labels = prepare_lstm_dataframe(shadow_inputs, shadow_labels,shadow_train_predictions,shadow_train_labels,shadow_train_losses,shadow_train_lstm,label = 1,include_losses = True, include_labels = True, include_lstm = True)
                shadow_inputs, shadow_labels = prepare_lstm_dataframe(shadow_inputs, shadow_labels,shadow_test_predictions,shadow_test_labels,shadow_test_losses,shadow_test_lstm,label = 0,include_losses = True, include_labels = True, include_lstm = True)
            else:
                shadow_train_predictions,shadow_train_labels,shadow_train_losses,shadow_train_prediction_labels = get_attack_features(shadow_trainloader,lstm)
                shadow_test_predictions,shadow_test_labels,shadow_test_losses,shadow_test_prediction_labels = get_attack_features(shadow_testloader,lstm)
                shadow_inputs, shadow_labels = [], []
                shadow_inputs, shadow_labels = prepare_dataframe(shadow_inputs, shadow_labels,shadow_train_predictions,shadow_train_labels,shadow_train_losses,label = 1,include_losses = True, include_labels = True)
                shadow_inputs, shadow_labels = prepare_dataframe(shadow_inputs, shadow_labels,shadow_test_predictions,shadow_test_labels,shadow_test_losses,label = 0,include_losses = True, include_labels = True)
            shadow_dataframe = create_dataframe(name, shadow_inputs, shadow_labels)
            shadow_d = create_statistics_dataframe(name,shadow_train_prediction_labels, shadow_train_labels, shadow_test_prediction_labels, shadow_test_labels)


