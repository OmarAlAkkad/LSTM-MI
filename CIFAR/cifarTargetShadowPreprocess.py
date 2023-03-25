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

    test_inputs = test[0]
    test_labels = test[1]

    return train_inputs, train_labels, test_inputs, test_labels

def split_target_shadow(X,y,split):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = split, stratify=y)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    train_inputs, train_labels, test_inputs, test_labels = create_dataset()

    split = 0.5
    target_train_inputs, shadow_train_inputs, target_train_labels, shadow_train_labels = split_target_shadow(train_inputs,train_labels,split)
    target_test_inputs, shadow_test_inputs, target_test_labels, shadow_test_labels = split_target_shadow(test_inputs,test_labels,split)

    pickle.dump(target_train_inputs, open('target_train_inputs.p', 'wb'))
    pickle.dump(shadow_train_inputs, open('shadow_train_inputs.p', 'wb'))
    pickle.dump(target_train_labels, open('target_train_labels.p', 'wb'))
    pickle.dump(shadow_train_labels, open('shadow_train_labels.p', 'wb'))
    pickle.dump(target_test_inputs, open('target_test_inputs.p', 'wb'))
    pickle.dump(shadow_test_inputs, open('shadow_test_inputs.p', 'wb'))
    pickle.dump(target_test_labels, open('target_test_labels.p', 'wb'))
    pickle.dump(shadow_test_labels, open('shadow_test_labels.p', 'wb'))
















