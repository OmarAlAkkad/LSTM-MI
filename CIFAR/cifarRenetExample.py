# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:23:59 2023

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
from model import build_model
import os
import cv2
import random

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

def prepare_sets(inputs, labels,number_of_classes):
    #this function is used to process the data into usable format.
    #convert inputs to float type and normalize to to range 0,1
    #inputs = inputs.reshape(-1,1)
    #scaler = StandardScaler()
    #inputs = scaler.fit_transform(inputs, labels)
    #inputs = inputs.reshape(-1,32,32,3)
    inputs = inputs.astype('float32') /255.0
    inputs = inputs.reshape(-1,32,32,3)
    #Let images have the shape (..., 1)
    #inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, stratify=labels, test_size=0.2, random_state=42)

    return x_train, y_train , x_test, y_test

def augment_data(inputs, labels):
    flipped = []
    for image in inputs:
        chance = random.randint(0,3)
        if chance == 0:
            flipped.append(cv2.flip(image,1))
        elif chance == 1:
            flipped.append(cv2.flip(image,0))
        else:
            flipped.append(image)

    right = np.float32([[1, 0, 2], [0, 1, 0]])
    left = np.float32([[1, 0, -2], [0, 1, 0]])
    up = np.float32([[1, 0, 0], [0, 1, -2]])
    down = np.float32([[1, 0, 0], [0, 1, 2]])

    shifted = []
    for image in flipped:
        chance = random.randint(0,3)
        if chance == 0:
            shifted.append(cv2.warpAffine(image, right, (image.shape[1], image.shape[0])))
        elif chance == 1:
            shifted.append(cv2.warpAffine(image, left, (image.shape[1], image.shape[0])))
        else:
            shifted.append(image)

    shifted2 = []
    for image in shifted:
        chance = random.randint(0,3)
        if chance == 0:
            shifted2.append(cv2.warpAffine(image, up, (image.shape[1], image.shape[0])))
        elif chance == 1:
            shifted2.append(cv2.warpAffine(image, down, (image.shape[1], image.shape[0])))
        else:
            shifted2.append(image)

    new_train = np.append(shifted2,inputs).reshape(-1,32,32,3)
    new_labels = np.append(labels, labels).reshape(-1,1)

    return np.array(new_train), np.array(new_labels)

if __name__ == "__main__":
    number = 10
    inputs, labels = create_dataset()

    inputs1, labels1 = augment_data(inputs, labels)

    x_train, y_train, x_test, y_test = prepare_sets(inputs1, labels1, number)

    model = build_model(10, 32, 32, 60)
    model.compile(loss='categorical_crossentropy',optimizer= 'adam' ,metrics=['accuracy'])
    checkpoint_path = "training_example/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history=model.fit(x_train,y_train,batch_size=60,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])

