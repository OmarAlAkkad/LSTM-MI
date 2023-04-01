# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:45:26 2023

@author: omars
"""
import pickle
import numpy as np
import tensorflow as tf
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

def load_data():
    data_file = open('mnist_target_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('mnist_target_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    return data, labels


def build_model(train, num_classes):
    model = Sequential()
    inputs = keras.Input(shape=((train.shape[1], train.shape[2])))
    lstm = (LSTM(512, activation="tanh",return_sequences=True))(inputs)
    lstm2 = (LSTM(512, activation = 'tanh',go_backwards=True, return_sequences = True))(lstm)
    flatten = layers.Flatten()(lstm2)
    dense1 = layers.Dense(256, activation="relu")(flatten)
    dense2 = layers.Dense(128, activation="relu")(dense1)
    dense3 = layers.Dense(64, activation="relu")(dense2)
    dense4 = layers.Dense(32, activation="relu")(dense3)
    outputs = layers.Dense(num_classes, activation = 'softmax')(dense4)
    model = keras.Model(inputs=inputs, outputs=outputs, name="target_mnist")
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

    model = build_model(x_train, num_classes)
    model.summary()

    batch_size = 100
    epochs = 20

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        validation_data = (x_test, y_test))


    score = model.evaluate(x_test, y_test)
    score_t = model.evaluate(x_train, y_train)
    print()
    print('Test loss:', score[0])
    print('Test accuracy : ', score[1])
    error_rate = round(1 - score[1], 3)

    print('error rate of :', error_rate)

    train_accuracy = score_t[1]
    test_accuracy = score[1]


    model.save('target_mnist')

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
    attention_weights = []

    for i in range(len(train_predictions)):
        # if train_predictions_labels[i] == y_train_label[i]:
            train_predictions_list[i].append(train_losses[i])
            train_predictions_list[i].append(y_train_label[i])
            inputs.append(train_predictions_list[i])
            all_labels.append(1)
    for i in range(len(test_predictions)):
        # if test_predictions_labels[i] == y_test_label[i]:
            test_predictions_list[i].append(test_losses[i])
            test_predictions_list[i].append(y_test_label[i])
            inputs.append(test_predictions_list[i])
            all_labels.append(0)

    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs,
        'Labels': all_labels
        }
    locals()['target_model_dataframe_mnist'] = pd.DataFrame(data=d)

    pickle.dump(locals()['target_model_dataframe_mnist'], open('target_renet_dataframe_mnist.p', 'wb'))

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
        locals()[f'precision_{set_type}'] = precision_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='macro')
        locals()[f'recall_{set_type}'] = recall_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='macro')

        locals()[f'f1_{set_type}'] = f1_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'], average='macro')
        models.append(f'model')
        Data.append(set_type)
        Accuracy.append(locals()[f'{set_type}_accuracy'])
        Precision.append(locals()[f'precision_{set_type}'])
        Recall.append(locals()[f'recall_{set_type}'])
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
    d.to_csv(f'target_mnist.csv')

    plot_data(history)






