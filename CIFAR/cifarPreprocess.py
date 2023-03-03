# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:24:52 2023

@author: omars
"""
import pickle
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_dataset():
    #load mnist dataset from keras.
    train, test = tf.keras.datasets.cifar10.load_data()

    x = train[0]
    y = train[1]

    x_ = test[0]
    y_ = test[1]

    inputs = []
    for i in range(len(y)):
        inputs.append(x[i])

    for i in range(len(y_)):
        inputs.append(x_[i])

    inputs = np.array(inputs)

    labels = []
    for i in range(len(y)):
        labels.append(y[i])

    for i in range(len(y_)):
        labels.append(y_[i])

    labels = np.array(labels)

    return inputs, labels

def split_target_shadow(X,y,split):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = split, stratify=y)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    inputs, labels = create_dataset()

    split = 0.5
    target_inputs, shadow_inputs, target_labels, shadow_labels = split_target_shadow(inputs,labels,split)

    pickle.dump(target_inputs, open('cifar_target_data.p', 'wb'))
    pickle.dump(target_labels, open('cifar_target_labels.p', 'wb'))
    pickle.dump(shadow_inputs, open('cifar_shadow_data.p', 'wb'))
    pickle.dump(shadow_labels, open('cifar_shadow_labels.p', 'wb'))
















