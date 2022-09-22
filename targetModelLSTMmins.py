# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:23:54 2022

@author: omars
"""
# -*- coding: utf-8 -*-
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, TimeDistributed
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
        x.append(data[i:(i + seq_size), 0:7])
        y.append(data[i:i+seq_size, 0])
    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 7))
    y = np.reshape(y, (y.shape[0], seq_size, 1))
    return x, y

def build_model(train):
    # train = train.reshape(-1, train.shape[1], train.shape[2], 1)
    model = Sequential()
    model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same", input_shape=(train.shape[1], train.shape[2])))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same"))
    model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # cnn.add(Flatten())
    model.add(TimeDistributed(Dense(32)))
    model.add(LSTM(64, activation = 'tanh',return_sequences = False))
    model.add(Dense(32))
    model.add(Dense(60))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

def rmse(predicted, actual):
    rmse = (np.sqrt(mean_squared_error(actual[:,0], predicted[:,0])))/statistics.mean(actual[:,0])

    return rmse

if __name__ == "__main__":
    dataset = load_data()
    dataset = np.reshape(dataset, (-1,7))
    scaler = pickle.load(open('scaler.p', 'rb'))
    scaler_predict = pickle.load(open('scaler_predictions.p', 'rb'))

    percentages = [0.7]
    epochs = [2]
    models = []
    train_rmse = []
    test_rmse = []

    for percent in percentages:
        for epoch in epochs:
            train, test = create_train_test(dataset, percent)

            past_mins = 60
            x_train, y_train = to_sequences(train, past_mins)
            x_test, y_test = to_sequences(test, past_mins)

            model = build_model(x_train)
            model.fit(x_train, y_train, epochs = epoch, validation_data = (x_test, y_test), verbose =1,batch_size = 1000)
            model.summary()
            model.save(f'target_{percent}_{epoch}')

            train_predictions = model.predict(x_train)
            test_predictions = model.predict(x_test)

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

            pickle.dump(locals()[f'target_model_dataframe_{percent}_{epoch}'], open(f'target_model_dataframe_{percent}_{epoch}.p', 'wb'))

            train_predictions = scaler_predict.inverse_transform(train_predictions)
            test_predictions = scaler_predict.inverse_transform(test_predictions)
            y_train = scaler_predict.inverse_transform(y_train)
            y_test = scaler_predict.inverse_transform(y_test)

            locals()[f'train_score_{percent}_{epoch}'] = rmse(train_predictions, y_train)
            locals()[f'test_score_{percent}_{epoch}'] = rmse(test_predictions, y_test)

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











