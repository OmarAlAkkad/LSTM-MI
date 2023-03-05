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

if __name__ == "__main__":
    number = 10
    inputs, labels = create_dataset()

    x_train, y_train, x_test, y_test = prepare_sets(inputs, labels, number)

    model = build_model(10, 32, 32, 60)
    model.compile(loss='categorical_crossentropy',optimizer= 'adam' ,metrics=['accuracy'])
    checkpoint_path = "training_example/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history=model.fit(x_train,y_train,batch_size=60,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])

