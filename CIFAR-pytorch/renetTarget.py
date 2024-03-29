# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:48:50 2023

@author: omars
"""
#coding:utf-8
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import sys
from torch.autograd import gradcheck
import time
import math
import argparse
import pickle

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import train,test
from transform import Relabel, ToLabel, Colorize

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = True
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



receptive_filter_size = 4
hidden_size = 320
image_size_w = 32
image_size_h = 32
batch_size = 50


# renet with one layer
class ReNet(nn.Module):
    def __init__(self, cuda, receptive_filter_size, hidden_size, batch_size, image_patches_height, image_patches_width):

        super(ReNet, self).__init__()

        self.batch_size = batch_size
        self.receptive_filter_size = receptive_filter_size
        self.input_size1 = receptive_filter_size * receptive_filter_size * 3
        self.input_size2 = hidden_size * 2
        self.hidden_size = hidden_size

		# vertical rnns
        self.rnn1 = nn.LSTM(self.input_size1, self.hidden_size, dropout = 0.2)
        self.rnn2 = nn.LSTM(self.input_size1, self.hidden_size, dropout = 0.2)

		# horizontal rnns
        self.rnn3 = nn.LSTM(self.input_size2, self.hidden_size, dropout = 0.2)
        self.rnn4 = nn.LSTM(self.input_size2, self.hidden_size, dropout = 0.2)

        self.initHidden(cuda)

        feature_map_dim = int(image_patches_height*image_patches_height*hidden_size*2)
        self.conv1 = nn.Conv2d(hidden_size*2, 2, 3,padding=1)#[1,640,8,8]->[1,1,8,8]
        self.UpsamplingBilinear2d=nn.UpsamplingBilinear2d(size=(32,32), scale_factor=None)
        self.dense = nn.Linear(feature_map_dim, 4096)
        self.fc = nn.Linear(4096, 10)

        self.log_softmax = nn.LogSoftmax()

    def initHidden(self, cuda):
        if cuda:
            self.hidden = ((torch.zeros(1, self.batch_size, self.hidden_size)).cuda(), (torch.zeros(1, self.batch_size, self.hidden_size)).cuda())
        else:
            self.hidden = ((torch.zeros(1, self.batch_size, self.hidden_size)), (torch.zeros(1, self.batch_size, self.hidden_size)))


    def get_image_patches(self, X, receptive_filter_size):
        """
		creates image patches based on the dimension of a receptive filter
		"""
        image_patches = []

        _, X_channel, X_height, X_width= X.size()


        for i in range(0, X_height, receptive_filter_size):
            for j in range(0, X_width, receptive_filter_size):
                X_patch = X[:, :, i: i + receptive_filter_size, j : j + receptive_filter_size]
                image_patches.append(X_patch)

        image_patches_height = (X_height // receptive_filter_size)
        image_patches_width = (X_width // receptive_filter_size)

        image_patches = torch.stack(image_patches)
        image_patches = image_patches.permute(1, 0, 2, 3, 4)

        image_patches = image_patches.contiguous().view(-1, image_patches_height, image_patches_width, receptive_filter_size * receptive_filter_size * X_channel)

        return image_patches



    def get_vertical_rnn_inputs(self, image_patches, forward):
        """
		creates vertical rnn inputs in dimensions
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
        vertical_rnn_inputs = []
        _, image_patches_height, image_patches_width, feature_dim = image_patches.size()

        if forward:
            for i in range(image_patches_height):
                for j in range(image_patches_width):
                    vertical_rnn_inputs.append(image_patches[:, j, i, :])

        else:
            for i in range(image_patches_height-1, -1, -1):
                for j in range(image_patches_width-1, -1, -1):
                    vertical_rnn_inputs.append(image_patches[:, j, i, :])

        vertical_rnn_inputs = torch.stack(vertical_rnn_inputs)


        return vertical_rnn_inputs



    def get_horizontal_rnn_inputs(self, vertical_feature_map, image_patches_height, image_patches_width, forward):
        """
		creates vertical rnn inputs in dimensions
		(num_patches, batch_size, rnn_input_feature_dim)
		num_patches: image_patches_height * image_patches_width
		"""
        horizontal_rnn_inputs = []

        if forward:
            for i in range(image_patches_height):
                for j in range(image_patches_width):
                    horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])
        else:
            for i in range(image_patches_height-1, -1, -1):
                for j in range(image_patches_width -1, -1, -1):
                    horizontal_rnn_inputs.append(vertical_feature_map[:, i, j, :])

        horizontal_rnn_inputs = torch.stack(horizontal_rnn_inputs)

        return horizontal_rnn_inputs


    def forward(self, X):

        """ReNet """

		# divide input input image to image patches
        image_patches = self.get_image_patches(X, self.receptive_filter_size)
        _, image_patches_height, image_patches_width, feature_dim = image_patches.size()

		# process vertical rnn inputs
        vertical_rnn_inputs_fw = self.get_vertical_rnn_inputs(image_patches, forward=True)
        vertical_rnn_inputs_rev = self.get_vertical_rnn_inputs(image_patches, forward=False)

		# extract vertical hidden states
        vertical_forward_hidden, vertical_forward_cell = self.rnn1(vertical_rnn_inputs_fw, self.hidden)
        vertical_reverse_hidden, vertical_reverse_cell = self.rnn2(vertical_rnn_inputs_rev, self.hidden)

		# create vertical feature map
        vertical_feature_map = torch.cat((vertical_forward_hidden, vertical_reverse_hidden), 2)
        vertical_feature_map =  vertical_feature_map.permute(1, 0, 2)

		# reshape vertical feature map to (batch size, image_patches_height, image_patches_width, hidden_size * 2)
        vertical_feature_map = vertical_feature_map.contiguous().view(-1, image_patches_width, image_patches_height, self.hidden_size * 2)
        vertical_feature_map.permute(0, 2, 1, 3)

		# process horizontal rnn inputs
        horizontal_rnn_inputs_fw = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height, image_patches_width, forward=True)
        horizontal_rnn_inputs_rev = self.get_horizontal_rnn_inputs(vertical_feature_map, image_patches_height, image_patches_width, forward=False)

		# extract horizontal hidden states
        horizontal_forward_hidden, horizontal_forward_cell = self.rnn3(horizontal_rnn_inputs_fw, self.hidden)
        horizontal_reverse_hidden, horizontal_reverse_cell = self.rnn4(horizontal_rnn_inputs_rev, self.hidden)

		# create horiztonal feature map[64,1,320]
        horizontal_feature_map = torch.cat((horizontal_forward_hidden, horizontal_reverse_hidden), 2)
        horizontal_feature_map =  horizontal_feature_map.permute(1, 0, 2)

		# flatten[1,64,640]
        output = horizontal_feature_map.contiguous().view(-1, image_patches_height * image_patches_width * self.hidden_size * 2)
        # output=output.permute(0,3,1,2)#[1,640,8,8]
# 		conv1=self.conv1(output)
# 		Upsampling=self.UpsamplingBilinear2d(conv1)
		# dense layer
        output = F.relu(self.dense(output))

# 		fully connected layer
        logits = self.fc(output)

		# log softmax
        logits = self.log_softmax(logits)

        return logits


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    s = '%s' % (asMinutes(s))
    return s

def load_data():
    data_file = open('target_dataset.p', 'rb')
    target_dataset = pickle.load(data_file)
    data_file.close()

    data_file = open('test_dataset.p', 'rb')
    test_dataset = pickle.load(data_file)
    data_file.close()

    return target_dataset, test_dataset



if __name__ == "__main__":

    target_dataset, test_dataset = load_data()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainloader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    renet = ReNet(args.cuda,receptive_filter_size, hidden_size, batch_size, image_size_w/receptive_filter_size, image_size_h/receptive_filter_size)
    if args.cuda:
        renet.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(renet.parameters(), lr=0.001)
    # optimizer = optim.SGD(renet.parameters(), lr=0.01, momentum = 0.5)

    for epoch in range(100):
        print('epoch:', epoch)

        running_loss = 0.0
        start = time.time()

        for i, data in enumerate(trainloader, 0):
			# get the inputs
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

			# wrap them in Variable
# 			inputs, labels = Variable(inputs), Variable(labels)

			# # zero the parameter gradients
            optimizer.zero_grad()
			# # forward + backward + optimize
            renet.initHidden(args.cuda)
            renet.train()
            logits = renet(inputs).to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

			# get statistics
            running_loss += loss.data
            _, predicted = torch.max(logits.data, 1)
            total = labels.size(0)
            correct = (predicted == labels.data).sum()
            train_accur = correct / total * 100


            eval_every = 10

			# print necessary info
            if i % eval_every == eval_every-1:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f time: %s' % (epoch + 1, i + 1, running_loss/(i + 1), timeSince(start)))
                print("train accuracy", train_accur)

        test_correct = 0
        test_total = 0
        for test_data in testloader:
            test_images, test_labels = test_data
            if args.cuda:
                test_images, test_labels = test_images.cuda(), test_labels.cuda()
            # calculate outputs by running images through the network
            test_outputs = renet(test_images)
            # the class with the highest energy is what we choose as prediction
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * test_correct // test_total} %')


    PATH = './cifar_renet.pth'
    torch.save(renet.state_dict(), PATH)

    renet_test = ReNet(args.cuda,receptive_filter_size, hidden_size, batch_size, image_size_w/receptive_filter_size, image_size_h/receptive_filter_size)
    if args.cuda:
        renet_test.cuda()

    renet_test.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = renet_test(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


