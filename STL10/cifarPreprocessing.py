# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:01:33 2023

@author: omars
"""
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset

if __name__ == "__main__":
    transform = transforms.Compose(
        [
         transforms.Resize((32,32)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)


    target_data, x_test, target_labels, y_test = train_test_split(trainset.data, trainset.targets, stratify=trainset.targets, test_size=0.8, random_state=42)
    shadow_data, _, shadow_labels, _ = train_test_split(x_test, y_test, stratify=y_test, test_size=0.75, random_state=42)
    x_test, y_test = testset.data, testset.targets
    target_data = target_data.astype('float32') /255.0
    target_data = target_data.reshape(-1,32,32,3)
    shadow_data = shadow_data.astype('float32') /255.0
    shadow_data = shadow_data.reshape(-1,32,32,3)
    x_test = x_test.astype('float32') /255.0
    x_test = x_test.reshape(-1,32,32,3)
    target_data, shadow_data, target_labels, shadow_labels = torch.from_numpy(target_data), torch.from_numpy(shadow_data), torch.Tensor(target_labels).long(), torch.Tensor(shadow_labels).long()
    x_test, y_test = torch.from_numpy(x_test), torch.Tensor(y_test).long()
    target_data = target_data.permute(0,3,1,2)
    shadow_data = shadow_data.permute(0,3,1,2)
    x_test = x_test.permute(0,3,1,2)
    target_dataset = torch.utils.data.TensorDataset(target_data, target_labels)
    shadow_dataset = torch.utils.data.TensorDataset(shadow_data, shadow_labels)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    pickle.dump(target_dataset, open('target_dataset.p', 'wb'))
    pickle.dump(shadow_dataset, open('shadow_dataset.p', 'wb'))
    pickle.dump(test_dataset, open('test_dataset.p', 'wb'))