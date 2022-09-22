# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 20:20:03 2022

@author: omars
"""
# load and clean-up data
import numpy as np
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, BatchNormalization
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from sklearn.utils import shuffle

def preprocess_dataset():
    # load all data
    dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
    # mark all missing values
    dataset.replace('?', nan, inplace=True)
    # make dataset numeric
    dataset = dataset.astype('float32')
    # fill missing
    fill_missing(dataset.values)
    # add a column for for the remainder of sub metering
    values = dataset.values
    dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])

    # save updated dataset
    dataset.to_csv('household_power_consumption.csv')
    daily_groups = dataset.resample('D')
    daily_data = daily_groups.sum()
    daily_data.to_csv('household_power_consumption_days.csv')
    label_days(daily_data)

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

def label_days(dataset):
    dataset = dataset.drop(dataset.index[[0,1]], axis=0)
    i = 0
    j = 0
    label = np.ones(len(dataset['Global_active_power']))
    while j < len(dataset['Global_active_power']):
        if i > 6:
            i = 0
        if i == 5 or i == 6:
            label[j] = 0
        j += 1
        i += 1
    dataset['labels'] = label
    dataset.to_csv('household_power_consumption_days_labelled.csv')


def load_dataset():
    dataset = pd.read_csv('household_power_consumption_days_labelled.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    return dataset

def plot_dataset(dataset):
    x = dataset['Global_active_power']
    y = range(0,len(x))
    labels = dataset['labels']
    colors = []
    for label in labels:
        if label == 1:
            colors.append('red')
        else:
            colors.append('blue')
    year = 1
    days = 365
    x1 = x[year * days : (year+1) * days]
    y1 = y[year * days : (year+1) * days]
    colors1 = colors[year * days : (year+1) * days]
    plt.bar(y1,x1,color = colors1)
    plt.show()

def train_test_split(dataset, years):
    total_years = 4
    days = 365
    train =  dataset[0 : (years) * days]
    test = dataset[(years) * days : total_years * days]
    train = shuffle(train)
    test = shuffle(test)
    x_train = train.drop(['labels'],axis = 1)
    y_train = train['labels']
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_train = np.array(y_train)
    x_test = test.drop(['labels'],axis = 1)
    y_test = test['labels']
    y_test = tf.keras.utils.to_categorical(y_test, 2)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test

def build_model(input_shape, number_of_classes):
    # train = train.reshape(-1, train.shape[1], train.shape[2], 1)
    model = Sequential() #initialize model
    model.add(tf.keras.Input(shape=(input_shape)))
    model.add(Flatten()) #flatten the array to input to dense layer
    model.add(BatchNormalization())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(2, activation='softmax')) #output layer with softmax activation function to get predictions vector
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model # return the created model

if __name__ == '__main__':
    preprocess_dataset()
    dataset = load_dataset()
    plot_dataset(dataset)
    years = 2
    number_of_classes = 2
    input_shape = (8)
    x_train, y_train, x_test, y_test = train_test_split(dataset, years)

    model = build_model(input_shape, number_of_classes)
    model.fit(x_train, y_train, epochs = 100, validation_data = (x_test, y_test), verbose =1)

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

    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []
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
      Recall.append(locals()[f'precision_{set_type}'])
      Negative_Recall.append( locals()[f'Negative_recall_{set_type}'])
      F1_Score.append(locals()[f'f1_{set_type}'])

    d = pd.DataFrame({'Model' : models,
        'Data': Data,
        'Accuracy': Accuracy,
        'Precision': Precision,
        'Recall': Recall,
        'Negative Recall': Negative_Recall,
        'F1 Score': F1_Score,})


    # train target model on weekdays, 24 hours to predict next 24 hours
    # test on weekends. weekdays labelled as 1 weekends labelled 0.
    # create shadow models same way. 50% split. maybe train on high consumption
    # months (june july august(summer), dec jan feb(winter))
    # after that check months.

    # cluster months to high energy and low energy..


















