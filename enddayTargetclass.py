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

def load_data():
    data_file = open('actual_model_data_class.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    datapoints = data.iloc[:,0:7]
    labels = data.iloc[:,8].to_numpy()
    months = data.iloc[:,9].to_numpy()

    scaler = StandardScaler()
    scaler = scaler.fit(datapoints)
    datapoints = scaler.transform(datapoints)

    return datapoints, labels, months, scaler

def create_train_test(dataset, labels, month, train_test_split):
    train_values = np.empty([0,7])
    test_values = np.empty([0,7])
    train_labels = np.empty([0,1])
    test_labels = np.empty([0,1])
    train_months = np.empty([0,1])
    test_months = np.empty([0,1])
    for i in range(int(len(dataset) * train_test_split)):
            train_values = np.vstack([train_values,dataset[i]])
            train_labels = np.vstack([train_labels,labels[i]])
            train_months = np.vstack([train_months,months[i]])

    for i in range(int(len(dataset) * train_test_split), len(dataset)):
            test_values = np.vstack([test_values,dataset[i]])
            test_labels = np.vstack([test_labels,labels[i]])
            test_months = np.vstack([test_months,months[i]])

    return train_values, test_values, train_labels, test_labels, train_months, test_months

def to_sequences(data, labels, month, seq_size):
    x = []
    y = []
    months = []

    for i in range(0,len(data)- 2*seq_size, seq_size):
        x.append(data[i:(i + seq_size), 0:7])
        y.append(labels[i])
        months.append(int(month[i]))

    x = np.array(x)
    y = np.array(y)
    months = np.array(months)
    x = np.reshape(x, (x.shape[0], x.shape[1], 7))
    y = np.reshape(y, (y.shape[0],1))
    months = np.reshape(months, (months.shape[0],1))
    y = tf.keras.utils.to_categorical(y)
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
    model.add(LSTM(128, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(32, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    data, labels, months, scaler = load_data()

    split = 0.5

    train_values, test_values, train_labels, test_labels, train_months, test_months = create_train_test(data, labels, months, split)

    models = []
    train_rmse = []
    test_rmse = []

    past_hours = 24
    x_train, y_train, training_months = to_sequences(train_values, train_labels, train_months, past_hours)
    x_test, y_test, testing_months = to_sequences(test_values, test_labels, test_months, past_hours)

    model = build_model(x_train)
    model.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test), verbose =1)
    model.save('target_enddays_class')

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

    inputs = []
    for i in range(len(train_predictions)):
        inputs.append(train_predictions[i])
    for i in range(len(test_predictions)):
        inputs.append(test_predictions[i])

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
    locals()['target_model_dataframe_enddays'] = pd.DataFrame(data=d)

    pickle.dump(locals()['target_model_dataframe_enddays'], open('target_model_dataframe_enddays.p', 'wb'))


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

    test_accuracy = model.evaluate(x_test, y_test)[1]
    train_accuracy = model.evaluate(x_train, y_train)[1]

    correct_months_train = []
    correct_months_test = []
    for i in range(len(y_test)):
        if y_test_label[i] == test_predictions_labels[i]:
            correct_months_test.append(int(testing_months[i][0]))
        if y_train_label[i] == train_predictions_labels[i]:
            correct_months_train.append(int(training_months[i][0]))

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
    d.to_csv(f'target_enddays_class.csv')

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
        train_months_percentage.append(correct_months_train.count(i)/training_months.tolist().count(i))
        test_months_percentage.append(correct_months_test.count(i)/testing_months.tolist().count(i))

    plt.figure()
    plt.title('Training set success rate')
    plt.bar(number_of_months,train_months_percentage)
    plt.figure()
    plt.title('Testing set success rate')
    plt.bar(number_of_months,test_months_percentage)





