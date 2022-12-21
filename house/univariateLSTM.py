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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import statistics
from sklearn.metrics import mean_squared_error

def load_data():
    data_file = open('actual_model_data_scaled.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    return data

# def normalize_dataset(data):
#   scaler = MinMaxScaler(feature_range = (0,1))
#   scaled_data = scaler.fit_transform(data)

#   return scaled_data

# def denormalize_dataset(data, descale):
#   scaler = MinMaxScaler(feature_range = (0,1))
#   scaled_data = scaler.fit_transform(data)
#   descaled_data = scaler.inverse_transform(descale)

#   return descaled_data

def create_train_test(data, percentage):
    train = data[0:int(percentage*len(data))]
    test = data[int(percentage*len(data)):]

    return train, test

def to_sequences(data, seq_size):
    x = []
    y = []

    for i in range(len(data) - seq_size):
        x.append(data[i:(i + seq_size), 0])
        y.append(data[i+seq_size, 0])

    return np.array(x), np.array(y)

def build_model(train):
    model = Sequential()
    model.add(LSTM(512, activation = 'relu',return_sequences = True, input_shape = (train.shape[1], train.shape[2])))
    model.add(LSTM(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

def rmse(predicted, actual):
    rmse = (np.sqrt(mean_squared_error(actual[:,0], predicted[:,0])))/statistics.mean(actual[:,0])

    return rmse

if __name__ == "__main__":
    dataset = load_data()
    dataset = np.reshape(dataset, (-1,1))
    scaler = pickle.load(open('scaler.p', 'rb'))

    percentages = [0.5, 0.6, 0.7, 0.8]
    epochs = [10,25,50,100]
    models = []
    train_rmse = []
    test_rmse = []

    for percent in percentages:
        for epoch in epochs:
            train, test = create_train_test(dataset, percent)

            past_days = 7
            x_train, y_train = to_sequences(train, past_days)
            x_test, y_test = to_sequences(test, past_days)

            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            model = build_model(x_train)
            model.fit(x_train, y_train, epochs = epoch, validation_data = (x_test, y_test), verbose =1)
            model.save(f'target_{percent}_{epoch}')

            train_predictions = model.predict(x_train)
            test_predictions = model.predict(x_test)

            y_train = np.reshape(y_train, (-1,1))
            y_test = np.reshape(y_test, (-1,1))

            labels_train = []
            for x in range(len(x_train)):
              labels_train.append(1)

            labels_test = []
            for x in range(len(x_test)):
              labels_test.append(0)

            all_labels = []
            for x in range(len(labels_train)):
                all_labels.append(1)
            for x in range(len(labels_test)):
                all_labels.append(0)

            inputs = []
            for i in range(len(train_predictions)):
                inputs.append(x_train[i])
                inputs[i] = np.append(inputs[i],train_predictions[i])
            for i in range(len(test_predictions)):
                inputs.append(x_test[i])
                inputs[i + len(train_predictions)] = np.append(inputs[i + len(train_predictions)],test_predictions[i])

            temp = list(zip(inputs, all_labels))
            random.shuffle(temp)
            inputs, all_labels = zip(*temp)
            inputs, all_labels = list(inputs), list(all_labels)
            d = {
             'Inputs': inputs ,
             'Labels': all_labels
            }
            locals()[f'target_model_dataframe_{percent}_{epoch}'] = pd.DataFrame(data=d)


            train_predictions = scaler.inverse_transform(train_predictions)
            test_predictions = scaler.inverse_transform(test_predictions)
            y_train = scaler.inverse_transform(y_train)
            y_test = scaler.inverse_transform(y_test)

            locals()[f'train_score_{percent}_{epoch}'] = rmse(train_predictions, y_train)
            locals()[f'test_score_{percent}_{epoch}'] = rmse(test_predictions, y_test)

            pickle.dump(locals()[f'target_model_dataframe_{percent}_{epoch}'], open(f'target_model_dataframe_{percent}_{epoch}.p', 'wb'))

            models.append(f'Model_{percent}_{epoch}')
            train_rmse.append(locals()[f'train_score_{percent}_{epoch}'])
            test_rmse.append(locals()[f'test_score_{percent}_{epoch}'])

    df = pd.DataFrame(
        {'Model': models, 'Train RMSE': train_rmse, 'Test RMSE':test_rmse}
    )

    df.to_csv('Target Model Train and Test RMSE.csv')


    # true_values = []
    # for x in range(len(y_train)):
    #     true_values.append(y_train[x][0])
    # for x in range(len(y_test)):
    #     true_values.append(y_test[x][0])

    # predicted_values = []
    # for x in range(len(train_predictions)):
    #     predicted_values.append(train_predictions[x][0])
    # for x in range(len(test_predictions)):
    #     predicted_values.append(test_predictions[x][0])

    # d = {
    #  'True Values': true_values ,
    #  'Predicted Values': predicted_values,
    #  'Labels': all_labels
    # }
    # target_model_dataframe = pd.DataFrame(data=d)











