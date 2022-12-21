# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:24:02 2022

@author: omars
"""
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
from sklearn.preprocessing import StandardScaler

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
    daily_groups = dataset.resample('H')
    daily_data = daily_groups.sum()
    daily_data = daily_data.drop(daily_data.index[0:31], axis=0)
    daily_data = daily_data.drop(daily_data.index[34536:], axis=0)
    daily_data.to_csv('household_power_consumption_hours.csv')

    return daily_data

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

def label_days(dataset):
    i = 0
    j = 0
    label = np.ones(len(dataset['Global_active_power']))
    while j < len(dataset['Global_active_power']):
        if i >= 24*7:
            i = 0
        if 24*5 <= i < 24*7:
            label[j] = 0
        j += 1
        i += 1

    return label

def load_dataset():
    dataset = pd.read_csv('household_power_consumption_hours.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    return dataset

def number_of_years(data):

  minutes_in_a_year = 365 * 24

  years = int(round(len(data) / minutes_in_a_year))

  return years


def split_csv_yearly(data, year):

    minutes_in_a_year = 360 * 24

    datapoints = data.iloc[year * minutes_in_a_year : (year+1) * minutes_in_a_year,:]

    return datapoints

def split_data_yearly(data, year):

    minutes_in_a_year = 360 * 24

    datapoints = data[year * minutes_in_a_year : (year+1) * minutes_in_a_year]

    return datapoints

def create_train_test_datasets(dataset,labels):
    train = dataset
    test = dataset
    for i in range(len(labels)):
        if labels[i] == 1:
            test = test.drop(dataset.index[i], axis=0)
        else:
            train = train.drop(dataset.index[i], axis=0)
    return train, test

def create_train_test_datasets_scaled(dataset,labels):
    train = np.empty([0,8])
    test = np.empty([0,8])
    for i in range(len(labels)):
        if labels[i] > 0:
            train = np.vstack([train,dataset[i]])
        else:
            test = np.vstack([test,dataset[i]])
    return train, test


if __name__ == '__main__':
    try:
        dataset = load_dataset()
    except:
        dataset = preprocess_dataset()

    labels = label_days(dataset)
    years = number_of_years(dataset)
    for year in range(years):
        locals()[f'dataset_year_{year}'] = split_csv_yearly(dataset, year)
        locals()[f'labels_year_{year}'] = labels[:len(locals()[f'dataset_year_{year}'])]

    actual_model = [dataset_year_0, dataset_year_1]
    actual_model_data = pd.concat(actual_model)
    actual_model_labels = np.concatenate((labels_year_0, labels_year_1), axis=0)

    shadow_model = [dataset_year_2, dataset_year_3]
    shadow_model_data = pd.concat(shadow_model)
    shadow_model_labels = np.concatenate((labels_year_2, labels_year_3), axis=0)

    pickle.dump(actual_model_data, open('actual_model_data.p', 'wb'))
    pickle.dump(actual_model_labels, open('actual_model_labels.p', 'wb'))
    pickle.dump(shadow_model_data, open('shadow_model_data.p', 'wb'))
    pickle.dump(shadow_model_labels, open('shadow_model_labels.p', 'wb'))

    target_train, target_test = create_train_test_datasets(actual_model_data,actual_model_labels)
    shadow_train, shadow_test = create_train_test_datasets(shadow_model_data,shadow_model_labels)

    pickle.dump(target_train, open('actual_model_data_train_u.p', 'wb'))
    pickle.dump(target_test, open('actual_model_data_test_u.p', 'wb'))
    pickle.dump(shadow_train, open('shadow_model_data_train_u.p', 'wb'))
    pickle.dump(shadow_test, open('shadow_model_data_test_u.p', 'wb'))

    scaler = StandardScaler()
    scaler.fit(dataset)
    dataset1 = scaler.transform(dataset)
    pickle.dump(scaler, open('scaler.p', 'wb'))

    years = number_of_years(dataset1)
    for year in range(years):
        locals()[f'dataset_year_{year}'] = split_data_yearly(dataset1, year)

    actual_model_data = np.concatenate((dataset_year_0, dataset_year_1), axis=0)

    shadow_model_data = np.concatenate((dataset_year_2, dataset_year_3), axis=0)

    target_train, target_test = create_train_test_datasets_scaled(actual_model_data,actual_model_labels)
    shadow_train, shadow_test = create_train_test_datasets_scaled(shadow_model_data,shadow_model_labels)

    pickle.dump(target_train, open('actual_model_data_train.p', 'wb'))
    pickle.dump(target_test, open('actual_model_data_test.p', 'wb'))
    pickle.dump(shadow_train, open('shadow_model_data_train.p', 'wb'))
    pickle.dump(shadow_test, open('shadow_model_data_test.p', 'wb'))



# def create_shadow_target_datasets(train, test):
#     target_train = train.iloc[0:int(len(train['labels'])*0.5),:]
#     target_test = test.iloc[0:int(len(test['labels'])*0.5),:]
#     shadow_train = train.iloc[int(len(train['labels'])*0.5):len(train['labels']),:]
#     shadow_test = test.iloc[int(len(test['labels'])*0.5):len(test['labels']),:]
#     return target_train, target_test, shadow_train, shadow_test



# def create_shadow_target_datasets_scaled(train, test):
#     target_train = np.empty([0,9])
#     target_test = np.empty([0,9])
#     shadow_train = np.empty([0,9])
#     shadow_test = np.empty([0,9])
#     for i in range(0,int(len(train)*0.5)):
#         target_train = np.vstack([target_train,train[i]])
#     for i in range(0,int(len(test)*0.5)):
#         target_test = np.vstack([target_test,test[i]])
#     for i in range(int(len(train)*0.5),len(train)):
#         shadow_train = np.vstack([shadow_train,train[i]])
#     for i in range(int(len(test)*0.5),len(test)):
#         shadow_test = np.vstack([shadow_test,test[i]])

#     return target_train, target_test, shadow_train, shadow_test


