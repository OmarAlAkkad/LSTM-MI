# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:45:26 2023

@author: omars
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Conv1D, MaxPool1D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, TimeDistributed, Attention
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import keras
from keras import models, layers
from keras import backend as K
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from keras_self_attention import SeqSelfAttention
from lstm_model import build_model
import os

def load_data():
    data_file = open('mnist_target_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('mnist_target_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    return data, labels

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

def prepare_sets(inputs, labels,number_of_classes):
    #this function is used to process the data into usable format.
    #convert inputs to float type and normalize to to range 0,1
    inputs = inputs.astype("float32") / 255.0
    #Let images have the shape (..., 1)
    # inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, stratify=labels, test_size=0.50, random_state=42)

    return x_train, y_train , x_test, y_test

def get_attention_weights(weight):
    averaged = []
    for x in range(weight.shape[0]):
        datapoint = []
        for i in range(weight.shape[2]):
            total = 0
            for j in range(weight.shape[1]):
                total += weight[x][j][i]
            total /= weight.shape[1]
            datapoint.append(total)
        averaged.append(datapoint)
    return averaged


if __name__ == "__main__":
    inputs, labels = load_data()

    num_classes = 10

    x_train, y_train , x_test, y_test = prepare_sets(inputs, labels, num_classes)

    models = []
    train_rmse = []
    test_rmse = []

    opt = Adam(learning_rate = 0.0001)

    model = build_model(num_classes)
    model.compile(loss='categorical_crossentropy',optimizer= opt ,metrics=['accuracy'])

    checkpoint_path = "training_lstm_target/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

    history=model.fit(x_train,y_train, batch_size=512 ,epochs=100, validation_data = (x_test, y_test), callbacks=[cp_callback])
    model.summary()

    score = model.evaluate(x_test, y_test)
    score_t = model.evaluate(x_train, y_train)
    print()
    print('Test loss:', score[0])
    print('Test accuracy : ', score[1])
    error_rate = round(1 - score[1], 3)

    print('error rate of :', error_rate)


    plot_data(history)






