# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:32:23 2023

@author: omars
"""
from model import build_model
from tensorflow.keras.optimizers import Adam
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
import random
import keras.backend as K
import os

def load_data():
    data_file = open('mnist_target_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('mnist_target_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    return data, labels

def prepare_sets(inputs, labels,number_of_classes):
    #this function is used to process the data into usable format.
    #convert inputs to float type and normalize to to range 0,1
    inputs = inputs.astype("float32") / 255.0
    inputs = inputs.reshape(-1,28,28,1)
    #Let images have the shape (..., 1)
    # inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, stratify=labels, test_size=0.5, random_state=42)

    return x_train, y_train , x_test, y_test


def plot_data(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    inputs, labels = load_data()

    num_classes = 10

    x_train, y_train , x_test, y_test = prepare_sets(inputs, labels, num_classes)

    opt = Adam(lr=0.0001)

    model = build_model(10, 28, 28,250)
    model.compile(loss='categorical_crossentropy',optimizer= opt ,metrics=['accuracy'])

    checkpoint_path = "training_target/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

    history=model.fit(x_train,y_train, batch_size=250 ,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])


    print('Train loss:', history.history['loss'])
    print('Train accuracy : ', history.history['accuracy'])
    print('Test loss:', history.history['val_loss'])
    print('Test accuracy : ', history.history['val_accuracy'])
    error_rate = round(1 - history.history['val_accuracy'][0], 3)
    print('error rate of :', error_rate)
