# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:21:24 2022

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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def create_csv_minutely():
    # load all data
    dataset = read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
    # mark all missing values
    dataset.replace('?', nan, inplace=True)
    dataset = dataset.dropna()
    # make dataset numeric
    dataset = dataset.astype('float32')
    # add a column for for the remainder of sub metering
    # save updated dataset
    dataset.to_csv('household_power_consumption.csv')

def create_client_dataset(dataset, number_of_clients, client):

    points_num = int(round(len(dataset)/number_of_clients))

    client_dataset = dataset[client * points_num : (client+1) * points_num,:]

    return client_dataset

def number_of_years(data):

  minutes_in_a_year = 365 * 24 * 60

  years = int(round(len(data) / minutes_in_a_year))

  return years

def split_data_yearly(data, year):

    minutes_in_a_year = 360 * 24 * 60

    datapoints = data[year * minutes_in_a_year : (year+1) * minutes_in_a_year]

    return datapoints

def split_csv_yearly(data, year):

    minutes_in_a_year = 360 * 24 * 60

    datapoints = data.iloc[year * minutes_in_a_year : (year+1) * minutes_in_a_year,:]

    return datapoints

if __name__ == '__main__':
    dataset = pd.read_csv('household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

    years = number_of_years(dataset)
    for year in range(years):
        locals()[f'dataset_year_{year}'] = split_data_yearly(dataset, year)
        pickle.dump(locals()[f'dataset_year_{year}'], open(f'dataset_year_{year}_unscaled.p', 'wb'))

    actual_model = [dataset_year_0, dataset_year_1]
    actual_model_data = pd.concat(actual_model)

    shadow_model = [dataset_year_2, dataset_year_3]
    shadow_model_data = pd.concat(shadow_model)

    pickle.dump(actual_model_data, open('actual_model_data_unscaled.p', 'wb'))
    pickle.dump(shadow_model_data, open('shadow_model_data_unscaled.p', 'wb'))

    scaler = StandardScaler()
    scaler.fit(dataset)
    dataset1 = scaler.transform(dataset)
    pickle.dump(scaler, open('scaler.p', 'wb'))

    predictions = dataset["Global_active_power"]
    predictions = predictions.to_numpy().reshape(-1,1)
    scaler_predict = StandardScaler()
    scaler_predict.fit(predictions)
    predictions = scaler_predict.transform(predictions)
    pickle.dump(scaler, open('scaler_predictions.p', 'wb'))

    for year in range(years):
        locals()[f'dataset_year_{year}'] = split_data_yearly(dataset1, year)
        pickle.dump(locals()[f'dataset_year_{year}'], open(f'dataset_year_{year}_scaled.p', 'wb'))

    # fl_training_dataset = dataset_year_0.append(dataset_year_1)
    # clients = 6
    # for client in range(clients):
    #     locals()[f'dataset_client_{client}'] = create_client_dataset(fl_training_dataset, clients, client)
    #     x_file_train = open(f'dataset_client_{client}.p', 'wb')
    #     pickle.dump(locals()[f'dataset_client_{client}'], x_file_train)
    #     x_file_train.close()

    actual_model_data = np.concatenate((dataset_year_0, dataset_year_1), axis=0)

    shadow_model_data = np.concatenate((dataset_year_2, dataset_year_3), axis=0)


    pickle.dump(actual_model_data, open('actual_model_data_scaled.p', 'wb'))
    pickle.dump(shadow_model_data, open('shadow_model_data_scaled.p', 'wb'))

