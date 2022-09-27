# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 13:40:30 2022

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
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from sklearn.utils import resample

def load_data():
    data_file = open('shadow_model_data_class.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    datapoints = data.iloc[:,0:7]
    labels = data.iloc[:,8].to_numpy()
    months = data.iloc[:,9].to_numpy()

    scaler = StandardScaler()
    scaler = scaler.fit(datapoints)
    datapoints = scaler.transform(datapoints)

    return datapoints, labels, months, scaler

def create_train_test(dataset, labels, month, test_month):
    train_values = np.empty([0,7])
    test_values = np.empty([0,7])
    train_labels = np.empty([0,1])
    test_labels = np.empty([0,1])
    train_months = np.empty([0,1])
    test_months = np.empty([0,1])
    test_values_extended = np.empty([0,7])
    test_labels_extended = np.empty([0,1])
    test_months_extended = np.empty([0,1])
    for i in range(int(len(dataset))):
        if month[i] == test_month:
            test_values = np.vstack([test_values,dataset[i]])
            test_labels = np.vstack([test_labels,labels[i]])
            test_months = np.vstack([test_months,months[i]])
        else:
            train_values = np.vstack([train_values,dataset[i]])
            train_labels = np.vstack([train_labels,labels[i]])
            train_months = np.vstack([train_months,months[i]])

    for i in range(int(len(train_labels)/len(test_labels))):
        test_values_extended = np.append(test_values_extended,test_values)
        test_labels_extended = np.append(test_labels_extended,test_labels)
        test_months_extended = np.append(test_months_extended,test_months)

    train_values = np.reshape(train_values, (-1, dataset.shape[1], 7))
    train_labels = np.reshape(train_labels, (-1,1))
    train_months = np.reshape(train_months, (-1,1))
    test_values_extended = np.reshape(test_values_extended, (-1, dataset.shape[1], 7))
    test_labels_extended = np.reshape(test_labels_extended, (-1,1))
    test_months_extended = np.reshape(test_months_extended, (-1,1))
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels_extended = tf.keras.utils.to_categorical(test_labels_extended)

    return train_values, test_values_extended, train_labels, test_labels_extended, train_months, test_months_extended

def to_sequences(data, labels, month, seq_size):
    x = []
    y = []
    months = []

    for i in range(0,len(data)- 2*seq_size, seq_size):
        x.append(data[i:(i + seq_size), 0:7])
        y.append(labels[i])
        months.append(int(month[i]))

    temp = list(zip(x, y, months))
    random.shuffle(temp)
    x, y, months = zip(*temp)
    x = np.array(x)
    y = np.array(y)
    months = np.array(months)
    x = np.reshape(x, (x.shape[0], x.shape[1], 7))
    y = np.reshape(y, (y.shape[0],1))
    months = np.reshape(months, (months.shape[0],1))
    return x, y, months

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
    # model.add(LSTM(128, activation = 'tanh',return_sequences=True, input_shape = (train.shape[1], train.shape[2])))
    model.add(LSTM(64, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(32, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    test_months = np.arange(1,13)

    for month in test_months:
        print(f'training model: {month}')
        data, labels, months, scaler = load_data()
        past_hours = 24
        x_data, y_data, months = to_sequences(data, labels, months, past_hours)

        x_train, x_test, y_train, y_test, training_months, testing_months = create_train_test(x_data, y_data, months, month)

        models = []
        train_rmse = []
        test_rmse = []

        model = build_model(x_train)
        model.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test), verbose =1)
        model.save(f'{month}_shadow_enddays_class')

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)

        indices = []
        index = 0
        train_predictions_labels = []
        for prediction in train_predictions:
            if prediction[0] > prediction [1]:
                train_predictions_labels.append(0)
                indices.append(index)
                index += 1
            else:
                train_predictions_labels.append(1)
                indices.append(index)
                index += 1

        test_predictions_labels = []
        for prediction in test_predictions:
            if prediction[0] > prediction [1]:
                test_predictions_labels.append(0)
                indices.append(index)
                index += 1
            else:
                test_predictions_labels.append(1)
                indices.append(index)
                index += 1

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

        train_losses = tf.keras.backend.categorical_crossentropy(y_train, train_predictions).numpy()
        test_losses = tf.keras.backend.categorical_crossentropy(y_test, test_predictions).numpy()

        train_predictions_list = train_predictions.tolist()
        test_predictions_list = test_predictions.tolist()

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

        correct_months_train = []
        correct_months_test = []
        inputs = []
        for i in range(len(y_test_label)):
            if y_test_label[i] == test_predictions_labels[i]:
                correct_months_test.append(int(testing_months[i][0]))
                inputs.append([test_predictions_list[i][0],test_predictions_list[i][1],train_losses[i]])
        for i in range(len(y_train_label)):
            if y_train_label[i] == train_predictions_labels[i]:
                correct_months_train.append(int(training_months[i][0]))
                inputs.append([train_predictions_list[i][0],train_predictions_list[i][1],train_losses[i]])

        filtered_months = np.append(training_months,testing_months)

        temp = list(zip(indices, inputs, filtered_months, all_labels))
        random.shuffle(temp)
        indices, inputs, filtered_months, all_labels = zip(*temp)
        indices, inputs, filtered_months,  all_labels = list(indices), list(inputs), list(filtered_months), list(all_labels)
        d = {
            'Index': indices,
            'Inputs': inputs ,
            'Months': filtered_months,
            'Labels': all_labels
            }
        locals()[f'{month}_shadow_model_dataframe_enddays'] = pd.DataFrame(data=d)

        pickle.dump(locals()[f'{month}_shadow_model_dataframe_enddays'], open(f'{month}_shadow_model_dataframe_enddays.p', 'wb'))

        test_accuracy = model.evaluate(x_test, y_test)[1]
        train_accuracy = model.evaluate(x_train, y_train)[1]

        correct_months_test.sort()
        correct_months_train.sort()

        sets = ["train", "test"]
        models = []
        Accuracy = []
        Precision = []
        Recall = []
        Negative_Recall = []
        F1_Score = []
        Data = []
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
              'F1 Score': F1_Score,
              })
        d.to_csv(f'{month}_shadow_enddays_class.csv')

        number_of_months = np.arange(1,13)
        correct_months_train_count = []
        correct_months_test_count = []
        count_train_months = []
        count_test_months = []
        train_months_percentage = []
        test_months_percentage = []

        for i in number_of_months:
            correct_months_train_count.append(correct_months_train.count(i))
            correct_months_test_count.append(correct_months_test.count(i))
            count_train_months.append(training_months.tolist().count(i))
            count_test_months.append(testing_months.tolist().count(i))
            try:
                train_months_percentage.append(correct_months_train.count(i)/training_months.tolist().count(i))
            except:
                train_months_percentage.append(0)
            try:
                test_months_percentage.append(correct_months_test.count(i)/testing_months.tolist().count(i))
            except:
                test_months_percentage.append(0)

        plt.figure()
        plt.title(f'{month} Shadow Model Training set success rate')
        plt.bar(number_of_months,train_months_percentage)
        plt.figure()
        plt.title(f'{month} Shadow Model Testing set success rate')
        plt.bar(number_of_months,test_months_percentage)
        plt.figure()
        plt.title(f'{month} Shadow Model Number of each month in training set')
        plt.bar(number_of_months,count_train_months)
        plt.figure()
        plt.title(f'{month} Shadow Model Number of each month in testing set')
        plt.bar(number_of_months,count_test_months)










