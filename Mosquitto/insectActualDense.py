# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:05:30 2022

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
import keras
from keras import models, layers
from keras import backend
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix

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

def build_model(train, input_shape, num_classes):
    # train = train.reshape(-1, train.shape[1], train.shape[2], 1)
    model = Sequential()
    # model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same", input_shape=(input_shape)))
    # model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same"))
    # model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # model.add(Flatten())
    # model.add(TimeDistributed(Dense(32)))
    # model.add(LSTM(1024, activation = 'relu', input_shape = (input_shape)))
    model.add(Dense(128, activation = 'relu', input_shape = (input_shape)))
    model.add(Dense(128, activation = 'tanh'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'sigmoid'))
    model.add(Dense(16, activation = 'sigmoid'))
    model.add(Dense(num_classes, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',  metrics=['accuracy'])

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

def prepare_sets(inputs, labels):
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, test_size=0.50, random_state=42)

    # x_train = np.array(feature_scaling_datasets(x_train))
    # x_test = np.array(feature_scaling_datasets(x_test))
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    num_classes = np.unique(y_train)
    num_classes = len(num_classes)
    input_shape = (x_train.shape[1], )
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return input_shape, num_classes, x_train, y_train , x_test, y_test


if __name__ == "__main__":
    inputs, labels = load_data()

    input_shape, num_classes, x_train, y_train , x_test, y_test = prepare_sets(inputs, labels)

    models = []
    train_rmse = []
    test_rmse = []

    model = build_model(x_train, input_shape, num_classes)
    model.summary()

    batch_size = 300
    epochs = 200

    model.fit(x_train, y_train,
              epochs = epochs,
              validation_data = (x_test, y_test),
              verbose =1, shuffle=True,
              batch_size = batch_size)

    score = model.evaluate(x_test, y_test)
    score_t = model.evaluate(x_train, y_train)
    print('Test loss:', score[0])
    print('Test accuracy : ', score[1])
    error_rate = round(1 - score[1], 3)

    print('error rate of :', error_rate)

    train_accuracy = score_t[1]
    test_accuracy = score[1]


    model.save('target_insect')

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

    train_losses = tf.keras.backend.categorical_crossentropy(y_train, train_predictions).numpy()
    test_losses = tf.keras.backend.categorical_crossentropy(y_test, test_predictions).numpy()

    train_predictions_list = train_predictions.tolist()
    test_predictions_list = test_predictions.tolist()

    inputs = []
    all_labels = []
    for i in range(len(train_predictions)):
        if train_predictions_labels[i] == y_train_label[i]:
            train_predictions_list[i].append(train_losses[i])
            train_predictions_list[i].append(y_train_label[i])
            inputs.append(train_predictions_list[i])
            all_labels.append(1)
    for i in range(len(test_predictions)):
        if test_predictions_labels[i] == y_test_label[i]:
            test_predictions_list[i].append(test_losses[i])
            test_predictions_list[i].append(y_test_label[i])
            inputs.append(test_predictions_list[i])
            all_labels.append(0)

    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs ,
        'Labels': all_labels
        }
    locals()['target_model_dataframe_insect'] = pd.DataFrame(data=d)

    pickle.dump(locals()['target_model_dataframe_insect'], open('target_model_dataframe_insect.p', 'wb'))

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
    d.to_csv(f'target_insect.csv')

    # plot_data(history)






