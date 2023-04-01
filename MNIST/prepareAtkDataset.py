# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 22:01:52 2023

@author: omars
"""
from model import build_model
from tensorflow.keras.optimizers import Adam, SGD
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix, accuracy_score
import random
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2

def load_data(data_type):
    print("loading data")
    data_file = open(f'mnist_{data_type}_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open(f'mnist_{data_type}_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    return data, labels

def prepare_sets(inputs, labels,number_of_classes):
    print("preparing sets")
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

def load_model(data_type):
    print("loading model")
    checkpoint_path = f"training_{data_type}/cp.ckpt"
    model = build_model(10, 32, 32, 1)

    model.load_weights(checkpoint_path)

    return model

def get_prediction_vectors(set_type,num_classes, model, data, labels):
    print(f"Getting prediction vector {set_type}")
    predictions = np.empty(0, dtype="float32")

    for x in range(len(data)):
        predictions = np.append(predictions, np.array(K.eval(model.predict(np.expand_dims(data[x], axis=0)))))

    predictions = predictions.reshape(-1,num_classes)

    predictions_labels = []
    for pred in predictions:
        predictions_labels.append(np.argmax(pred, axis=0))

    y_label = []
    for pred in labels:
        y_label.append(np.argmax(pred, axis=0))

    pickle.dump(predictions, open(f'{set_type}_predictions_mnist.p', 'wb'))

    return predictions, predictions_labels, y_label

def get_lstm_vectors(set_type,model, data, num_of_neurons, direction = 0):
    print(f"Getting LSTM vectors {set_type}")
    lstm = np.empty(0, dtype="float32")

    for x in range(len(data)):
        lstm = np.append(lstm, np.array(K.eval(model.lstm(np.expand_dims(data[x], axis=0))))[0][direction][0][:num_of_neurons])

    lstm = lstm.reshape(-1,num_of_neurons)

    pickle.dump(lstm, open(f'{set_type}_lstm_mnist.p', 'wb'))

    return lstm

def prepare_dataframe(inputs,all_labels,predictions,losses,lstm,labels,label = 1,include_losses = True, include_labels = True,include_lstm =True):
    print("Preparing data frame")
    for i in range(len(predictions)):
        if include_losses:
            predictions[i].append(losses[i])
        if include_labels:
            predictions[i].append(labels[i])
            if include_lstm:
                predictions[i].append(lstm[i])
        inputs.append(predictions[i])
        all_labels.append(label)

    return inputs, all_labels

def create_dataframe(data, inputs, all_labels):
    print("creating data frame")
    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs,
        'Labels': all_labels
        }
    locals()[f'{data}_renet_dataframe_mnist'] = pd.DataFrame(data=d)

    pickle.dump(locals()[f'{data}_renet_dataframe_mnist'], open(f'{data}_renet_dataframe_mnist.p', 'wb'))

    return locals()[f'{data}_renet_dataframe_mnist']

def create_statistics_dataframe(train_predictions_labels, y_train_label, test_predictions_labels, y_test_label):
    print("creating statistics dataframe")
    sets = ["train", "test"]
    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []
    for set_type in sets:
        locals()[f'{set_type}_accuracy'] = accuracy_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
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
    d.to_csv(f'{data}_renet_mnist.csv')

    return d

if __name__ == '__main__':
    datas = ['shadow','target']

    for data in datas:
        inputs, labels = load_data(data)

        num_classes = 10

        x_train, y_train , x_test, y_test = prepare_sets(inputs, labels, num_classes)

        model = load_model(data)

        train_predictions, train_predictions_labels, y_train_label = get_prediction_vectors(data,num_classes,model,x_train, y_train)
        test_predictions, test_predictions_labels, y_test_label = get_prediction_vectors(data,num_classes,model, x_test, y_test)

        train_losses = tf.keras.backend.categorical_crossentropy(y_train, train_predictions).numpy()
        test_losses = tf.keras.backend.categorical_crossentropy(y_test, test_predictions).numpy()

        train_lstm = get_lstm_vectors(data, model, x_train, num_of_neurons = 1280, direction = 0)
        test_lstm = get_lstm_vectors(data, model, x_test, num_of_neurons = 1280, direction = 0)

        train_predictions_list = train_predictions.tolist()
        test_predictions_list = test_predictions.tolist()

        inputs = []
        all_labels = []

        inputs, all_labels = prepare_dataframe(inputs,all_labels,train_predictions_list,train_losses,train_lstm,y_train_label,label = 1,include_losses = True, include_labels = True,include_lstm=True)
        inputs, all_labels = prepare_dataframe(inputs,all_labels,test_predictions_list,test_losses,test_lstm,y_test_label,label = 0,include_losses = True, include_labels = True,include_lstm=True)

        locals()[f'{data}_renet_dataframe_mnist'] = create_dataframe(data, inputs, all_labels)

        d = create_statistics_dataframe(train_predictions_labels, y_train_label, test_predictions_labels, y_test_label)
