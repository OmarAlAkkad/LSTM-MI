# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:42:44 2023

@author: omars
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def load_data(name):
    print(f'loading data for {name}')
    data_file = open(f'{name}-Shadow_dataframe.p', 'rb')
    shadow = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{name}-Target_dataframe.p', 'rb')
    target = pickle.load(data_file)
    data_file.close()

    shadow_labels = shadow['Labels'].to_numpy()
    shadow_inputs = shadow['Inputs'].to_numpy()

    target_labels = target['Labels'].to_numpy()
    target_inputs = target['Inputs'].to_numpy()

    return shadow_inputs, shadow_labels, target_inputs, target_labels

def create_dataframe(name,dataset, inputs, all_labels):
    print(f"creating data frame for {name}-{dataset}-Neurons_dataframe")
    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs,
        'Labels': all_labels
        }
    dataframe = pd.DataFrame(data=d)

    pickle.dump(dataframe, open(f'{name}-{dataset}-Neurons_dataframe.p', 'wb'))

    return dataframe

def preprocess_data(inputs, labels):
    print('preprocessing data')
    #this function is used to process the data into usable format.
    #Let images have the shape (..., 1)
    elements = len(inputs[0])
    inputs=np.array([np.array(xi) for xi in inputs])
    inputs = inputs.reshape(-1, elements)
    non_lstm_inputs = inputs[:,:12]
    inputs = inputs[:, 12:]

    labels = labels.reshape(-1,1)
    return inputs, non_lstm_inputs, labels

if __name__ == "__main__":
    models = [('DLA-BiLSTM'),
              ('DLA-LSTM'),
              ('ResNet18-BiLSTM'),
              ('ResNet18-LSTM'),
              ('DenseNet121-BiLSTM'),
              ('DenseNet121-LSTM'),
              ('VGG-BiLSTM'),
              ('VGG-LSTM'),
              ]

    models = [
              ('VGG-BiLSTM'),
              ('VGG-LSTM'),
              ]

    layers_scores = []
    layers_vals = []

    for method_name in models:
        print(f"Getting best LSTM neurons for {method_name}")

        shadow_inputs, shadow_labels, target_inputs, target_labels = load_data(method_name)

        shadow_inputs, shadow_non_lstm_inputs, shadow_labels = preprocess_data(shadow_inputs, shadow_labels)
        target_inputs, target_non_lstm_inputs, target_labels = preprocess_data(target_inputs, target_labels)

        x_train, x_test, y_train, y_test = train_test_split(shadow_inputs, shadow_labels, test_size=0.25, random_state=12)

        print('training random forest classifier')
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(x_train, y_train.ravel())
        accuracy = accuracy_score(y_test, rf.predict(x_test))

        print("LSTM Layer Accuracy " + str(accuracy))
        layers_scores.append(accuracy)

        print('getting important features')
        important_features = rf.feature_importances_
        index_list = [i for i in range(len(important_features))]

        df = pd.DataFrame({'Features': important_features, 'Index': index_list})

        df = df.sort_values(by='Features', ascending=False)

        indexes_to_sort_by = df['Index']

        shadow_inputs_sorted = np.take(shadow_inputs, indexes_to_sort_by, axis=1)
        target_inputs_sorted = np.take(target_inputs, indexes_to_sort_by, axis=1)

        shadow_new_inputs = np.concatenate((shadow_non_lstm_inputs, shadow_inputs_sorted), axis=1)
        target_new_inputs = np.concatenate((target_non_lstm_inputs, target_inputs_sorted), axis=1)

        shadow_df = create_dataframe(method_name,'Shadow', shadow_new_inputs, shadow_labels)
        target_df = create_dataframe(method_name,'Target', target_new_inputs,  target_labels)

