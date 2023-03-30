# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:57:18 2023

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
    data_file = open(f'{data_type}_train_inputs.p', 'rb')
    x_train = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{data_type}_test_inputs.p', 'rb')
    x_test = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{data_type}_train_labels.p', 'rb')
    y_train = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{data_type}_test_labels.p', 'rb')
    y_test = pickle.load(data_file)
    data_file.close()

    return x_train, x_test, y_train, y_test

def test_on_other_data():
    data_file = open('testing_images.p', 'rb')
    testing_images = pickle.load(data_file)
    data_file.close()

    data_file = open('testing_labels.p', 'rb')
    testing_labels = pickle.load(data_file)
    data_file.close()

    data_file = open('testing_images1.p', 'rb')
    testing_images1 = pickle.load(data_file)
    data_file.close()

    data_file = open('testing_labels1.p', 'rb')
    testing_labels1 = pickle.load(data_file)
    data_file.close()

    data_file = open('testing_images2.p', 'rb')
    testing_images2 = pickle.load(data_file)
    data_file.close()

    data_file = open('testing_labels2.p', 'rb')
    testing_labels2 = pickle.load(data_file)
    data_file.close()

    model = load_model(data)

    x_train, y_train = prepare_sets(testing_images, testing_labels, num_classes)

    train_predictions = np.empty(0, dtype="float32")

    for x in range(len(x_train)):
        train_predictions = np.append(train_predictions, np.array(K.eval(model.predict(np.expand_dims(x_train[x], axis=0)))))

    train_predictions = train_predictions.reshape(-1,num_classes)

    train_predictions_labels = []
    for pred in train_predictions:
        train_predictions_labels.append(np.argmax(pred, axis=0))

    y_train_label = []
    for pred in y_train:
        y_train_label.append(np.argmax(pred, axis=0))

    print("accuracy of testing set 0:",accuracy_score(y_train_label, train_predictions_labels))

    x_train, y_train = prepare_sets(testing_images1, testing_labels1, num_classes)

    train_predictions = np.empty(0, dtype="float32")

    for x in range(len(x_train)):
        train_predictions = np.append(train_predictions, np.array(K.eval(model.predict(np.expand_dims(x_train[x], axis=0)))))

    train_predictions = train_predictions.reshape(-1,num_classes)

    train_predictions_labels = []
    for pred in train_predictions:
        train_predictions_labels.append(np.argmax(pred, axis=0))

    y_train_label = []
    for pred in y_train:
        y_train_label.append(np.argmax(pred, axis=0))

    print("accuracy of testing set 1:",accuracy_score(y_train_label, train_predictions_labels))

    x_train, y_train = prepare_sets(testing_images2, testing_labels2, num_classes)

    train_predictions = np.empty(0, dtype="float32")

    for x in range(len(x_train)):
        train_predictions = np.append(train_predictions, np.array(K.eval(model.predict(np.expand_dims(x_train[x], axis=0)))))

    train_predictions = train_predictions.reshape(-1,num_classes)

    train_predictions_labels = []
    for pred in train_predictions:
        train_predictions_labels.append(np.argmax(pred, axis=0))

    y_train_label = []
    for pred in y_train:
        y_train_label.append(np.argmax(pred, axis=0))

    print("accuracy of testing set 2:",accuracy_score(y_train_label, train_predictions_labels))


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

    return np.array(inputs), np.array(labels)

def load_model(data_type):
    checkpoint_path = f"training_{data_type}/cp.ckpt"
    model = build_model(10, 32, 32, 1)

    model.load_weights(checkpoint_path)

    return model

if __name__ == '__main__':
    data = 'shadow'
    x_train, x_test, y_train, y_test = load_data(data)

    num_classes = 10

    # x_train, y_train = augment_images(x_train, y_train, num_classes)
    x_train, y_train = prepare_sets(x_train, y_train, num_classes)
    x_test, y_test = prepare_sets(x_test, y_test, num_classes)

    model = load_model(data)

    train_predictions = np.empty(0, dtype="float32")
    test_predictions = np.empty(0, dtype= 'float32')

    for x in range(len(x_train)):
        train_predictions = np.append(train_predictions, np.array(K.eval(model.predict(np.expand_dims(x_train[x], axis=0)))))

    for x in range(len(x_test)):
        test_predictions = np.append(test_predictions, np.array(K.eval(model.predict(np.expand_dims(x_test[x], axis=0)))))

    train_predictions = train_predictions.reshape(-1,num_classes)
    test_predictions = test_predictions.reshape(-1,num_classes)

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
    locals()[f'{data}_renet_dataframe_cifar'] = pd.DataFrame(data=d)

    pickle.dump(locals()[f'{data}_renet_dataframe_cifar'], open(f'{data}_renet_dataframe_cifar.p', 'wb'))

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
    d.to_csv(f'{data}_renet_cifar.csv')

    test_on_other_data()