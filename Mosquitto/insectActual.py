# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:54:38 2022

@author: omars
"""
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, TimeDistributed, Attention
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from Grid import *
import keras
from keras import models, layers
from keras import backend
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from lstm_model import build_model
import os

def load_data():
    data_file = open('insect_target_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('insect_target_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    return data, labels

def feature_scaling_datasets(ts_datasets):
    normalized_ts_datasets = []

    for ts in ts_datasets:
        normalized_ts_datasets.append(feature_scaling(ts))

    return normalized_ts_datasets

def feature_scaling(ts):
    n = len(ts)
    maximum = max(ts)
    minimum = min(ts)

    normalized_ts = list.copy(ts)
    r = maximum-minimum
    for i in range(n):
        normalized_ts[i] = (ts[i]-minimum)/r

    return normalized_ts

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

def prepare_sets(inputs, labels, m, n):
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, stratify=labels, test_size=0.50, random_state=42)

    x_train = feature_scaling_datasets(x_train)
    x_test = feature_scaling_datasets(x_test)

    g = Grid(m, n)
    x_train = g.dataset2Matrices(x_train)
    x_test = g.dataset2Matrices(x_test)

    img_rows, img_cols = x_train.shape[1:]
    class_set = set(y_train)

    print('class_set :', class_set)
    print('img_rows :', img_rows, 'img_columns :', img_cols)

    num_classes = len(class_set)
    min_class = min(class_set)

    y_train = [y-min_class for y in y_train]
    y_test = [y-min_class for y in y_test]

    if backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1,
                                  img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1,
                                img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0],
                                  img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows,
                                img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    #x_train /= 255
    #x_test /= 255

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    pickle.dump(x_train, open('x_train_insect_target_lstm.p', 'wb'))
    pickle.dump(y_train, open('y_train_insect_target_lstm.p', 'wb'))
    pickle.dump(x_test, open('x_test_insect_target_lstm.p', 'wb'))
    pickle.dump(y_test, open('y_test_insect_target_lstm.p', 'wb'))

    return input_shape, num_classes, x_train, y_train , x_test, y_test


if __name__ == "__main__":
    inputs, labels = load_data()

    m = 28
    n = 28

    input_shape, num_classes, x_train, y_train , x_test, y_test = prepare_sets(inputs, labels, m ,n)

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






