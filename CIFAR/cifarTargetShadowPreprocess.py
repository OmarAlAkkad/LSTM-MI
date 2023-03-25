# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:26:54 2023

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
    train_inputs = train[0]
    train_labels = train[1]
    target_x_train, x_test, target_y_train, y_test = train_test_split(train_inputs, train_labels, test_size = 0.8, stratify=train_labels)
    shadow_x_train, x_test, shadow_y_train, y_test = train_test_split(x_test, y_test, test_size = 0.75, stratify=y_test)

    test_inputs = test[0]
    test_labels = test[1]

    return target_x_train, target_y_train, shadow_x_train, shadow_y_train, test_inputs, test_labels

def split_target_shadow(X,y,split):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = split, stratify=y)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    target_x_train, target_y_train, shadow_x_train, shadow_y_train, test_inputs, test_labels = create_dataset()

    pickle.dump(target_x_train, open('target_train_inputs.p', 'wb'))
    pickle.dump(shadow_x_train, open('shadow_train_inputs.p', 'wb'))
    pickle.dump(target_y_train, open('target_train_labels.p', 'wb'))
    pickle.dump(shadow_y_train, open('shadow_train_labels.p', 'wb'))
    pickle.dump(test_inputs, open('target_test_inputs.p', 'wb'))
    pickle.dump(test_inputs, open('shadow_test_inputs.p', 'wb'))
    pickle.dump(test_labels, open('target_test_labels.p', 'wb'))
    pickle.dump(test_labels, open('shadow_test_labels.p', 'wb'))
















