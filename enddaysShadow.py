# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:37:24 2022

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
from sklearn.preprocessing import StandardScaler

def load_data():
    data_file = open('shadow_model_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('shadow_model_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    all_data = np.append(data, labels.reshape(-1,1), 1)

    scaler = StandardScaler()
    scaler = scaler.fit(all_data)
    all_data = scaler.transform(all_data)

    train, test = create_train_test(all_data)

    return train, test, scaler

def create_train_test(dataset):
    train = np.empty([0,9])
    test = np.empty([0,9])
    for i in range(len(dataset)):
        if dataset[i][8] > 0:
            train = np.vstack([train,dataset[i]])
        else:
            test = np.vstack([test,dataset[i]])
    return train, test

def to_sequences(data, seq_size):
    x = []
    y = []

    for i in range(len(data)- 2*seq_size):
        x.append(data[i:(i + seq_size), 0:7])
        y.append(data[i+seq_size:(i + 2*seq_size), 0])

    x = np.array(x)
    y = np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 7))
    y = np.reshape(y, (y.shape[0], seq_size,1))
    return x, y

def build_model(train):
    # train = train.reshape(-1, train.shape[1], train.shape[2], 1)
    model = Sequential()
    # model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same", input_shape=(train.shape[1], train.shape[2])))
    # model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # model.add(Conv1D(64, 2, strides=1, activation='relu', padding="same"))
    # model.add(MaxPool1D(pool_size=2, strides=2, padding='valid'))
    # # cnn.add(Flatten())
    # model.add(TimeDistributed(Dense(32)))
    model.add(LSTM(256, activation = 'relu',return_sequences=True, input_shape = (train.shape[1], train.shape[2])))
    model.add(LSTM(256, activation = 'relu'))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(24))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

def rmse(predicted, actual):
    predicted = np.array(predicted).reshape(-1,1)
    actual = np.array(actual).reshape(-1,1)
    rmse = (np.sqrt(mean_squared_error(actual[:,0], predicted[:,0])))/statistics.mean(actual[:,0])

    return rmse

def plot_data(predicted, actual,seq_size):
    # predicted = np.array(predicted).reshape(-1,1)
    # actual = np.array(actual).reshape(-1,1)
    predicted1 = []
    actual1 = []
    for i in range(0,len(predicted)- 2*seq_size, seq_size *seq_size):
        predicted1.append(predicted[i:(i + seq_size)])
        actual1.append(actual[i:(i + seq_size)])
    predicted1 = np.array(predicted1).reshape(-1,1)
    actual1 = np.array(actual1).reshape(-1,1)

    plt.figure()
    plt.plot(predicted1)
    plt.plot(actual1)
    plt.show()


if __name__ == "__main__":
    train, test, scaler = load_data()
    # scaler = pickle.load(open('scaler.p', 'rb'))

    models = []
    train_rmse = []
    test_rmse = []

    past_hours = 24
    x_train, y_train = to_sequences(train, past_hours)
    x_test, y_test = to_sequences(test, past_hours)

    model = build_model(x_train)
    model.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test), verbose =1,batch_size = 500)
    model.save('shadow_enddays')

    train_predictions = model.predict(x_train).reshape(-1,24,1)
    test_predictions = model.predict(x_test).reshape(-1,24,1)

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
        # inputs.append(train_predictions[i])
    for i in range(len(test_predictions)):
        inputs.append(x_test[i])
        inputs[i + len(train_predictions)] = np.append(inputs[i + len(train_predictions)],test_predictions[i])
        # inputs.append(test_predictions[i])

    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs ,
        'Labels': all_labels
        }
    locals()['shadow_model_dataframe_enddays'] = pd.DataFrame(data=d)

    pickle.dump(locals()['shadow_model_dataframe_enddays'], open('shadow_model_dataframe_enddays.p', 'wb'))

    descaled_train = []
    descaled_y_train = []
    for i in range(len(train_predictions)):
        descaled_train.append(scaler.inverse_transform(np.repeat(train_predictions[i], train.shape[1], axis = -1))[:,0])
        descaled_y_train.append(scaler.inverse_transform(np.repeat(y_train[i], train.shape[1], axis = -1))[:,0])
    descaled_test = []
    descaled_y_test = []
    for i in range(len(test_predictions)):
        descaled_test.append(scaler.inverse_transform(np.repeat(test_predictions[i], train.shape[1], axis = -1))[:,0])
        descaled_y_test.append(scaler.inverse_transform(np.repeat(y_test[i], train.shape[1], axis = -1))[:,0])

    locals()['train_score_enddays'] = rmse(descaled_train, descaled_y_train)
    locals()['test_score_enddays'] = rmse(descaled_test, descaled_y_test)

    models.append(f'Model_enddays')
    train_rmse.append(locals()[f'train_score_enddays'])
    test_rmse.append(locals()[f'test_score_enddays'])

    df = pd.DataFrame(
        {'Model': models, 'Train RMSE': train_rmse, 'Test RMSE':test_rmse}
    )

    df.to_csv('Shadow Model Train and Test RMSE enddays.csv')

    descaled_train = np.array(descaled_train).reshape(-1,1)
    descaled_y_train = np.array(descaled_y_train).reshape(-1,1)
    descaled_test = np.array(descaled_test).reshape(-1,1)
    descaled_y_test = np.array(descaled_y_test).reshape(-1,1)

    plot_data(descaled_train, descaled_y_train,past_hours)
    plot_data(descaled_test, descaled_y_test,past_hours)






