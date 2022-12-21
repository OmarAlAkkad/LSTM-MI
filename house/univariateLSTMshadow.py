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
    data_file = open('shadow_model_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    return data

def create_shadow_dataset(dataset, number_of_shadows, shadow):

    points_num = int(round(len(dataset)/number_of_shadows))

    shadow_dataset = dataset[shadow * points_num : (shadow+1) * points_num,:]

    return shadow_dataset

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
    shadow_models = 1
    past_days = 7

    percentages = [0.5, 0.6, 0.7, 0.8]
    epochs = [10,25,50,100]

    for percent in percentages:
        for epoch in epochs:
            inputs = []
            all_labels = []
            for shadow in range(0, shadow_models):
                locals()['shadow_' + str(shadow)] = create_shadow_dataset(dataset, shadow_models, shadow)
                locals()['train_' + str(shadow)], locals()['test_' + str(shadow)] = create_train_test(locals()['shadow_' + str(shadow)], percent)

                locals()['xtrain_' + str(shadow)], locals()['ytrain_' + str(shadow)] = to_sequences(locals()['train_' + str(shadow)], past_days)
                locals()['xtest_' + str(shadow)], locals()['ytest_' + str(shadow)] = to_sequences(locals()['test_' + str(shadow)], past_days)

                locals()['xtrain_' + str(shadow)] = np.reshape(locals()['xtrain_' + str(shadow)], (locals()['xtrain_' + str(shadow)].shape[0], locals()['xtrain_' + str(shadow)].shape[1], 1))
                locals()['xtest_' + str(shadow)] = np.reshape(locals()['xtest_' + str(shadow)], (locals()['xtest_' + str(shadow)].shape[0], locals()['xtest_' + str(shadow)].shape[1], 1))

                model = build_model(locals()['xtrain_' + str(shadow)])
                model.fit(locals()['xtrain_' + str(shadow)], locals()['ytrain_' + str(shadow)], epochs = epoch, validation_data = (locals()['xtest_' + str(shadow)], locals()['ytest_' + str(shadow)]), verbose =1)
                model.save(f'shadow_{shadow}_{percent}_{epoch}')

                locals()['train_predictions_' + str(shadow)] = model.predict(locals()['xtrain_' + str(shadow)])
                locals()['test_predictions_' + str(shadow)] = model.predict(locals()['xtest_' + str(shadow)])

                locals()['ytrain_' + str(shadow)] = np.reshape(locals()['ytrain_' + str(shadow)], (-1,1))
                locals()['ytest_' + str(shadow)] = np.reshape(locals()['ytest_' + str(shadow)], (-1,1))

                labels_train = []
                for x in range(len(locals()['xtrain_' + str(shadow)])):
                  labels_train.append(1)

                labels_test = []
                for x in range(len(locals()['xtest_' + str(shadow)])):
                  labels_test.append(0)

                for x in range(len(labels_train)):
                  all_labels.append(1)
                for x in range(len(labels_test)):
                  all_labels.append(0)

                locals()['inputs_' + str(shadow)] = []
                for i in range(len(locals()['train_predictions_' + str(shadow)])):
                     locals()['inputs_' + str(shadow)].append(locals()['xtrain_' + str(shadow)][i])
                     locals()['inputs_' + str(shadow)][i] = np.append(locals()['inputs_' + str(shadow)][i],locals()['train_predictions_' + str(shadow)][i])
                for i in range(len(locals()['test_predictions_' + str(shadow)])):
                     locals()['inputs_' + str(shadow)].append(locals()['xtest_' + str(shadow)][i])
                     locals()['inputs_' + str(shadow)][i + len(locals()['train_predictions_' + str(shadow)])] = np.append(locals()['inputs_' + str(shadow)][i + len(locals()['train_predictions_' + str(shadow)])],locals()['test_predictions_' + str(shadow)][i])

                inputs.extend(locals()['inputs_' + str(shadow)])

            temp = list(zip(inputs, all_labels))
            random.shuffle(temp)
            inputs, all_labels = zip(*temp)
            inputs, all_labels = list(inputs), list(all_labels)
            d = {
                  'Inputs': inputs ,
                  'Labels': all_labels
            }
            locals()[f'shadow_model_dataframe_{percent}_{epoch}'] = pd.DataFrame(data=d)

            pickle.dump(locals()[f'shadow_model_dataframe_{percent}_{epoch}'], open(f'shadow_model_dataframe_{percent}_{epoch}.p', 'wb'))

            locals()['train_predictions_' + str(shadow)] = scaler.inverse_transform(locals()['train_predictions_' + str(shadow)])
            locals()['test_predictions_' + str(shadow)] = scaler.inverse_transform(locals()['test_predictions_' + str(shadow)])
            locals()['ytrain_' + str(shadow)] = scaler.inverse_transform(locals()['ytrain_' + str(shadow)])
            locals()['ytest_' + str(shadow)] = scaler.inverse_transform(locals()['ytest_' + str(shadow)])

            locals()[f'train_score_{shadow}_{percent}_{epoch}'] = rmse(locals()['train_predictions_' + str(shadow)], locals()['ytrain_' + str(shadow)])
            locals()[f'test_score_{shadow}_{percent}_{epoch}'] = rmse(locals()['test_predictions_' + str(shadow)], locals()['ytest_' + str(shadow)])

    models = []
    train = []
    test = []
    for percent in percentages:
        for epoch in epochs:
            models.append(f'Shadow_model_{shadow}_{percent}_{epoch}')
            train.append(locals()[f'train_score_{shadow}_{percent}_{epoch}'])
            test.append(locals()[f'test_score_{shadow}_{percent}_{epoch}'] )

    df = pd.DataFrame(
        {'Shadow Model': models, 'Train RMSE': train, 'Test RMSE':test}
    )

    df.to_csv('Shadow Model Train and Test RMSE.csv')
    #     for x in range(len(locals()['ytrain_' + str(shadow)])):
    #         true_values.append(locals()['ytrain_' + str(shadow)][x][0])
    #     for x in range(len(locals()['ytest_' + str(shadow)])):
    #         true_values.append(locals()['ytest_' + str(shadow)][x][0])

    #     for x in range(len(locals()['train_predictions_' + str(shadow)])):
    #         predicted_values.append(locals()['train_predictions_' + str(shadow)][x][0])
    #     for x in range(len(locals()['test_predictions_' + str(shadow)])):
    #         predicted_values.append(locals()['test_predictions_' + str(shadow)][x][0])

    # d = {
    #  'True Values': true_values ,
    #  'Predicted Values': predicted_values,
    #  'Labels': all_labels
    # }
    # shadow_model_dataframe = pd.DataFrame(data=d)


















