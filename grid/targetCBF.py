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
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, TimeDistributed
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from Grid import *
import keras
from keras import models, layers
from keras import backend
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix

def load_data():
    data_file = open('CBF_actual_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('CBF_actual_labels.p', 'rb')
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

def build_model(train, input_shape, num_classes):
    # train = train.reshape(-1, train.shape[1], train.shape[2], 1)
    model = Sequential()
    # model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same", input_shape=(train.shape[1], train.shape[2])))
    # model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same"))
    # model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # # cnn.add(Flatten())
    # model.add(TimeDistributed(Dense(32)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
    model.add(layers.BatchNormalization(name="batch_norm_1"))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.BatchNormalization(name="batch_norm_2"))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer='rmsprop',
                     metrics=['accuracy'])

    return model

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
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, test_size=0.50, random_state=42)

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

    return input_shape, num_classes, x_train, y_train , x_test, y_test


if __name__ == "__main__":
    inputs, labels = load_data()

    m = 28
    n = 28

    input_shape, num_classes, x_train, y_train , x_test, y_test = prepare_sets(inputs, labels, m ,n)

    models = []
    train_rmse = []
    test_rmse = []

    model = build_model(x_train, input_shape, num_classes)
    model.summary()

    batch_size = 5
    epochs = 100

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_split=0.1)

    score = model.evaluate(x_test, y_test)
    score_t = model.evaluate(x_train, y_train)
    print()
    print('Test loss:', score[0])
    print('Test accuracy : ', score[1])
    error_rate = round(1 - score[1], 3)

    print('error rate of :', error_rate)

    train_accuracy = score_t[1]
    test_accuracy = score[1]


    model.save('target_CBF')

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    train_predictions_labels = []
    for pred in train_predictions:
        train_predictions_labels.append(np.argmax(pred, axis=0))


    test_predictions_labels = []
    for pred in test_predictions:
        test_predictions_labels.append(np.argmax(pred, axis=0))

    y_train_label = []
    for pred in y_train:
        y_train_label.append(np.argmax(pred, axis=0))

    y_test_label = []
    for pred in y_test:
        y_test_label.append(np.argmax(pred, axis=0))


    labels_train = []
    for x in range(len(x_train)):
        labels_train.append(1)

    labels_test = []
    for x in range(len(x_test)):
        labels_test.append(0)

    all_labels = []
    for x in range(len(labels_train)):
        all_labels.append(1)
    for x in range(len(labels_test)):
        all_labels.append(0)

    train_losses = tf.keras.backend.categorical_crossentropy(y_train, train_predictions).numpy()
    test_losses = tf.keras.backend.categorical_crossentropy(y_test, test_predictions).numpy()

    train_predictions_list = train_predictions.tolist()
    test_predictions_list = test_predictions.tolist()

    inputs = []
    for i in range(len(train_predictions)):
        train_predictions_list[i].append(train_losses[i])
        train_predictions_list[i].append(y_train_label[i])
        inputs.append(train_predictions_list[i])
    for i in range(len(test_predictions)):
        test_predictions_list[i].append(test_losses[i])
        test_predictions_list[i].append(y_test_label[i])
        inputs.append(test_predictions_list[i])

    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs ,
        'Labels': all_labels
        }
    locals()['target_model_dataframe_CBF'] = pd.DataFrame(data=d)

    pickle.dump(locals()['target_model_dataframe_CBF'], open('target_model_dataframe_CBF.p', 'wb'))

    sets = ["train", "test"]
    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []
    for set_type in sets:
        locals()[f'confusion_matrix_{set_type}'] = confusion_matrix(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
        locals()[f'TN_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][0][0]
        locals()[f'FP_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][0][1]
        locals()[f'FN_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][1][0]
        locals()[f'TP_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][1][1]
        locals()[f'Negative_recall_{set_type}'] = locals()[f'TN_{set_type}'] / (locals()[f'TN_{set_type}'] + locals()[f'FP_{set_type}'])
        locals()[f'precision_{set_type}'] = precision_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='micro')
        locals()[f'recall_{set_type}'] = recall_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='micro')

        locals()[f'f1_{set_type}'] = f1_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'], average='micro')
        models.append(f'model')
        Data.append(set_type)
        Accuracy.append(locals()[f'{set_type}_accuracy'])
        Precision.append(locals()[f'precision_{set_type}'])
        Recall.append(locals()[f'precision_{set_type}'])
        Negative_Recall.append( locals()[f'Negative_recall_{set_type}'])
        F1_Score.append(locals()[f'f1_{set_type}'])

    d = pd.DataFrame({'Model' : models,
         'Data': Data,
         'Accuracy': Accuracy,
         'Precision': Precision,
         'Recall': Recall,
         'Negative Recall': Negative_Recall,
         'F1 Score': F1_Score,
         })
    d.to_csv(f'target_CBF.csv')

    plot_data(history)






