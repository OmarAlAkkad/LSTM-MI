# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 21:23:52 2022
@author: omars
"""
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def create_client_dataset(dataset, number_of_clients, client):

    points_num = int(round(len(dataset)/number_of_clients))

    client_dataset = dataset[client * points_num : (client+1) * points_num,:]

    return client_dataset

def number_of_years(data):

  days_in_a_year = 365

  years = int(round(len(data) / days_in_a_year))

  return years

def split_data_yearly(data, year):

    days_in_a_year = 361

    datapoints = data[year * days_in_a_year : (year+1) * days_in_a_year]

    return datapoints

def split_csv_yearly(data, year):

    days_in_a_year = 361

    datapoints = data.iloc[year * days_in_a_year : (year+1) * days_in_a_year,:]

    return datapoints

if __name__ == '__main__':
    dataset = pd.read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    dataset = dataset['Global_active_power'].array.reshape(-1, 1)

    years = number_of_years(dataset)
    for year in range(years):
        locals()[f'dataset_year_{year}'] = split_data_yearly(dataset, year)
        pickle.dump(locals()[f'dataset_year_{year}'], open(f'dataset_year_{year}_unscaled.p', 'wb'))

    actual_model_data = []
    for x in range(len(dataset_year_0)):
        actual_model_data.append(dataset_year_0[x])
    for x in range(len(dataset_year_1)):
        actual_model_data.append(dataset_year_1[x])

    shadow_model_data = []
    for x in range(len(dataset_year_2)):
        shadow_model_data.append(dataset_year_2[x])
    for x in range(len(dataset_year_3)):
        shadow_model_data.append(dataset_year_3[x])
    pickle.dump(actual_model_data, open('actual_model_data_unscaled.p', 'wb'))
    pickle.dump(shadow_model_data, open('shadow_model_data_unscaled.p', 'wb'))

    scaler = StandardScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    pickle.dump(scaler, open('scaler.p', 'wb'))

    for year in range(years):
        locals()[f'dataset_year_{year}'] = split_data_yearly(dataset, year)
        pickle.dump(locals()[f'dataset_year_{year}'], open(f'dataset_year_{year}_scaled.p', 'wb'))

    # fl_training_dataset = dataset_year_0.append(dataset_year_1)
    # clients = 6
    # for client in range(clients):
    #     locals()[f'dataset_client_{client}'] = create_client_dataset(fl_training_dataset, clients, client)
    #     x_file_train = open(f'dataset_client_{client}.p', 'wb')
    #     pickle.dump(locals()[f'dataset_client_{client}'], x_file_train)
    #     x_file_train.close()


    # actual_model_data = dataset_year_0.append(dataset_year_1)
    # shadow_model_data = dataset_year_2.append(dataset_year_3)
    actual_model_data = []
    for x in range(len(dataset_year_0)):
        actual_model_data.append(dataset_year_0[x])
    for x in range(len(dataset_year_1)):
        actual_model_data.append(dataset_year_1[x])

    shadow_model_data = []
    for x in range(len(dataset_year_2)):
        shadow_model_data.append(dataset_year_2[x])
    for x in range(len(dataset_year_3)):
        shadow_model_data.append(dataset_year_3[x])
    pickle.dump(actual_model_data, open('actual_model_data_scaled.p', 'wb'))
    pickle.dump(shadow_model_data, open('shadow_model_data_scaled.p', 'wb'))


