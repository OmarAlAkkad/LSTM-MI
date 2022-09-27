# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 20:18:06 2022

@author: omars
"""
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

def load_data(month):
    data_file = open(f'{month}_target_model_dataframe_enddays.p', 'rb')
    actual = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{month}_shadow_model_dataframe_enddays.p', 'rb')
    shadow = pickle.load(data_file)
    data_file.close()

    indices_train = []
    x_train = []
    months_train = []
    for element in range(len(shadow['Inputs'])):
        x_train.append(shadow['Inputs'][element])
        indices_train.append(shadow['Index'][element])
        months_train.append(int(shadow['Months'][element]))

    # x_train = np.reshape(x_train,(-1,8))
    y_train = shadow['Labels'].to_numpy()

    indices_test = []
    x_test = []
    months_test = []
    for element in range(len(actual['Inputs'])):
        x_test.append(actual['Inputs'][element])
        indices_test.append(actual['Index'][element])
        months_test.append(int(actual['Months'][element]))

    # x_test = np.reshape(x_test,(-1,8))
    y_test = actual['Labels'].to_numpy()

    return x_train, y_train, x_test, y_test, indices_train, indices_test, months_train, months_test

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
    # model = Sequential()
    # model.add(LSTM(32, activation = 'tanh',return_sequences = False, input_shape = input_shape))
    # model.add(Dropout(0.2))
    # model.add(Dense(128))
    # model.add(Dense(64))
    # model.add(Dense(32))
    # model.add(Dense(number_of_classes))
    # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

    model = Sequential() #initialize model
    model.add(tf.keras.Input(shape=(input_shape)))
    model.add(Flatten()) #flatten the array to input to dense layer
    model.add(BatchNormalization())

    model.add(Dense(1000, activation='relu',  kernel_initializer='he_normal'))
    model.add(Dense(2, activation='softmax')) #output layer with softmax activation function to get predictions vector
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model # return the created model

if __name__ == "__main__":
    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []

    test_months = np.arange(1,13)

    for month in test_months:

        print(f"training model: {month}")
        x_train, y_train, x_test, y_test, indices_train, indices_test, months_train, months_test = load_data(month)
        x_train, y_train = preprocess_data(x_train, y_train, 2)
        x_test, y_test = preprocess_data(x_test, y_test, 2)

        number_of_classes = 2
        input_shape = (x_train.shape[1],x_train.shape[2])
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

        correct_indices_train = []
        correct_indices_test = []
        correct_months_train = []
        correct_months_test = []
        for i in range(len(test_predictions_labels)):
            if y_test_label[i] == test_predictions_labels[i]:
                correct_indices_test.append(indices_test[i])
                correct_months_test.append(months_test[i])
        for i in range(len(train_predictions_labels)):
            if y_train_label[i] == train_predictions_labels[i]:
                correct_indices_train.append(indices_train[i])
                correct_months_train.append(months_train[i])

        correct_indices_test.sort()
        correct_indices_train.sort()
        correct_months_test.sort()
        correct_months_train.sort()

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
        d.to_csv(f'{month}_attack_models_class.csv')

        y_points_train = []
        x_points_train = np.arange(717,1500)
        for point in x_points_train:
            if point in correct_indices_train:
                y_points_train.append(1)
            else:
                y_points_train.append(0)
        plt.bar(x_points_train,y_points_train)

        y_points_test = []
        x_points_test = np.arange(0,750)
        for point in x_points_test:
            if point in correct_indices_test:
                y_points_test.append(1)
            else:
                y_points_test.append(0)
        plt.bar(x_points_test,y_points_test)

        number_of_months = np.arange(1,13)
        correct_months_train_count = []
        correct_months_test_count = []
        count_train_months = []
        count_test_months = []
        train_months_percentage = []
        test_months_percentage = []

        for i in number_of_months:
            correct_months_train_count.append(correct_months_train.count(i))
            correct_months_test_count.append(correct_months_test.count(i))
            count_train_months.append(months_train.count(i))
            count_test_months.append(months_test.count(i))
            try:
                train_months_percentage.append(correct_months_train.count(i)/months_train.count(i))
            except:
                train_months_percentage.append(0)
            try:
                test_months_percentage.append(correct_months_test.count(i)/months_test.count(i))
            except:
                test_months_percentage.append(0)


        plt.figure()
        plt.title(f'{month} Shadow model success rate')
        plt.bar(number_of_months,train_months_percentage)
        plt.figure()
        plt.title(f'{month} Actual model success rate')
        plt.bar(number_of_months,test_months_percentage)



        # train_predictions = scaler.inverse_transform(train_predictions)
        # test_predictions = scaler.inverse_transform(test_predictions)
        # y_train = scaler.inverse_transform(y_train)
        # y_test = scaler.inverse_transform(y_test)

        # train_score = rmse(train_predictions, y_train)
        # test_score = rmse(test_predictions, y_test)
        # ks-2samp test

        # visualize train and test data. histogram.
        # every prediction hour - avg of 24 hours input.

    # look into train and test accuracy on each month alone
