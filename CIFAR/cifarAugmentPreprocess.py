# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:25:40 2023

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def augment_images(data, labels, number_of_classes):
    datagen = ImageDataGenerator(
    featurewise_center=True,
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(data)
    # fits the model on batches with real-time data augmentation:
    aug_images = datagen.flow((data,labels), batch_size=1)

    images = []
    labels = []
    for image in aug_images:
        if len(images) == 100000:
            break
        images.append(image[0])
        labels.append(image[1])

    images = np.array(images).reshape(-1,32,32,3)
    labels = np.array(labels).reshape(-1,1)

    # labels = tf.keras.utils.to_categorical(labels, number_of_classes)

    return images,labels

if __name__ == "__main__":
    inputs, labels = create_dataset()

    split = 0.5
    number_of_classes = 10
    target_inputs, shadow_inputs, target_labels, shadow_labels = split_target_shadow(inputs,labels,split)
    target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(target_inputs, target_labels, stratify=target_labels, test_size=0.5, random_state=42)
    shadow_x_train, shadow_x_test, shadow_y_train, shadow_y_test = train_test_split(shadow_inputs, shadow_labels, stratify=shadow_labels, test_size=0.5, random_state=42)
    target_x_train, target_y_train = augment_images(target_x_train, target_y_train, number_of_classes)
    shadow_x_train, shadow_y_train = augment_images(shadow_x_train, shadow_y_train, number_of_classes)

    pickle.dump(target_x_train, open('cifar_target_x_train.p', 'wb'))
    pickle.dump(target_x_test, open('cifar_target_x_test.p', 'wb'))
    pickle.dump(target_y_train, open('cifar_target_y_train.p', 'wb'))
    pickle.dump(target_y_test, open('cifar_target_y_test.p', 'wb'))
    pickle.dump(shadow_x_train, open('cifar_shadow_x_train.p', 'wb'))
    pickle.dump(shadow_x_test, open('cifar_shadow_x_test.p', 'wb'))
    pickle.dump(shadow_y_train, open('cifar_shadow_y_train.p', 'wb'))
    pickle.dump(shadow_y_test, open('cifar_shadow_y_test.p', 'wb'))
















