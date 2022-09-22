# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 13:31:40 2022

@author: omars
"""
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    X = pd.read_csv('household_power_consumption_days.csv')

    return X

def normalize_dataset(data):
  scaler = StandardScaler()
  scaler = scaler.fit(data)
  scaled_data = scaler.transform(data)

  return scaled_data

def prepare_training(data, future_days, past_days):
  x = []
  Y = []

  for i in range(past_days, len(data) - future_days + 1):
    x.append(data[i - past_days:i, 0]) # use 0:data.shape[1] for multivariate
    Y.append(data[i + future_days - 1: i + future_days, 0])

  x,Y = np.array(x), np.array(Y)

  return x,Y

def build_model(x,y):
    model = Sequential()
    model.add(LSTM(64, activation = 'relu', input_shape = (x.shape[1], 1), return_sequences = True ))
    model.add(LSTM(32, activation = 'relu', return_sequences = False))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer = 'adam', loss = 'mse')

    return model

def forecast(data, x, Y, days, dates, model):

    forecast_period_dates = pd.date_range(list(dates)[-1], periods = days, freq = '1d').tolist()
    forecast = model.predict(x[-days:])
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    forecast_copies = np.repeat(forecast, data.shape[1], axis = -1)
    y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

    return forecast_period_dates, y_pred_future

if __name__ == "__main__":
    dataset = load_data()
    train_dates = pd.to_datetime(dataset['datetime'])
    new_dataset = dataset.iloc[: , 1:]
    normalized_data = normalize_dataset(new_dataset)
    future_days = 1
    past_days = 7
    x, Y = prepare_training(normalized_data, future_days, past_days)
    model = build_model(x, Y)
    history = model.fit(x,Y, epochs = 20, batch_size = 16, validation_split = 0.1, verbose =1)

    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.legend()







