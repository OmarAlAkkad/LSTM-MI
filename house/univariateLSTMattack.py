# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:38:27 2022

@author: omars
"""
# -*- coding: utf-8 -*-
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, BatchNormalization
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix

def load_data(percent,epoch):
    data_file = open(f'target_model_dataframe_{percent}_{epoch}.p', 'rb')
    actual = pickle.load(data_file)
    data_file.close()

    data_file = open(f'shadow_model_dataframe_{percent}_{epoch}.p', 'rb')
    shadow = pickle.load(data_file)
    data_file.close()

    x_train = []
    for element in range(len(shadow['Inputs'])):
        x_train.append(shadow['Inputs'][element])

    x_train = np.reshape(x_train,(-1,8))
    y_train = shadow['Labels'].to_numpy()

    x_test = []
    for element in range(len(actual['Inputs'])):
        x_test.append(actual['Inputs'][element])

    x_test = np.reshape(x_test,(-1,8))
    y_test = actual['Labels'].to_numpy()

    return x_train, y_train, x_test, y_test

def preprocess_data(inputs, labels, number_of_classes):
    #this function is used to process the data into usable format.
    #Let images have the shape (..., 1)
    inputs = np.array(inputs)
    inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    labels = np.array(labels)
    return inputs, labels

def create_model(input_shape, number_of_classes):
    #This function creates a deep neural network model using tensorflow for the cifar10 dataset.
    model = Sequential() #initialize model
    model.add(tf.keras.Input(shape=(input_shape)))
    model.add(Flatten()) #flatten the array to input to dense layer
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(2, activation='softmax')) #output layer with softmax activation function to get predictions vector
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model # return the created model

if __name__ == "__main__":
    percentages = [0.5, 0.6, 0.7, 0.8]
    epochs = [10,25,50,100]

    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []

    for percent in percentages:
        for epoch in epochs:
            x_train, y_train, x_test, y_test = load_data(percent,epoch)
            x_train, y_train = preprocess_data(x_train, y_train, 2)
            x_test, y_test = preprocess_data(x_test, y_test, 2)

            number_of_classes = 2
            input_shape = (8,1)
            model = create_model(input_shape, number_of_classes)
            history = model.fit(x_train, y_train, epochs = 100, validation_data = (x_test, y_test), verbose =1)

            train_predictions = model.predict(x_train)
            train_predictions_labels = []
            for pred in train_predictions:
              if pred[1] > pred[0]:
                train_predictions_labels.append(1)
              else:
                train_predictions_labels.append(0)

            test_predictions = model.predict(x_test)
            test_predictions_labels = []
            for pred in test_predictions:
              if pred[1] > pred[0]:
                test_predictions_labels.append(1)
              else:
                test_predictions_labels.append(0)

            y_train_label = []
            for pred in y_train:
              if pred[1] > pred[0]:
                y_train_label.append(1)
              else:
                y_train_label.append(0)

            y_test_label = []
            for pred in y_test:
                if pred[1] > pred[0]:
                    y_test_label.append(1)
                else:
                    y_test_label.append(0)

            test_accuracy = model.evaluate(x_test, y_test)[1]
            train_accuracy = model.evaluate(x_train, y_train)[1]

            sets = ["train", "test"]
            for set_type in sets:
              locals()[f'confusion_matrix_{set_type}'] = confusion_matrix(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
              locals()[f'TN_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][0][0]
              locals()[f'FP_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][0][1]
              locals()[f'FN_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][1][0]
              locals()[f'TP_{set_type}'] = locals()[f'confusion_matrix_{set_type}'][1][1]
              locals()[f'Negative_recall_{set_type}'] = locals()[f'TN_{set_type}'] / (locals()[f'TN_{set_type}'] + locals()[f'FP_{set_type}'])
              locals()[f'precision_{set_type}'] = precision_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
              locals()[f'recall_{set_type}'] = recall_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])

              locals()[f'f1_{set_type}'] = f1_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'])
              models.append(f'model_{percent}_{epoch}')
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
        d.to_csv(f'attack_models.csv')


    # train_predictions = scaler.inverse_transform(train_predictions)
    # test_predictions = scaler.inverse_transform(test_predictions)
    # y_train = scaler.inverse_transform(y_train)
    # y_test = scaler.inverse_transform(y_test)

    # train_score = rmse(train_predictions, y_train)
    # test_score = rmse(test_predictions, y_test)
    # ks-2samp test









