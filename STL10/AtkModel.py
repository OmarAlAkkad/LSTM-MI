# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 18:01:39 2023

@author: omars
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:13:41 2023

@author: omars
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, BatchNormalization
import pandas as pd
#import xgboost
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from white_box_model import build_model

def load_data(name):
    data_file = open(f'{name}-Target_dataframe.p', 'rb')
    target = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{name}-Shadow_dataframe.p', 'rb')
    shadow = pickle.load(data_file)
    data_file.close()

    y_train = shadow['Labels'].to_numpy()
    x_train = shadow['Inputs'].to_numpy()

    y_test = target['Labels'].to_numpy()
    x_test = target['Inputs'].to_numpy()


    return x_train, y_train, x_test, y_test

def load_lstm_data(name):
    data_file = open(f'{name}-Target-Neurons_dataframe.p', 'rb')
    target = pickle.load(data_file)
    data_file.close()

    data_file = open(f'{name}-Shadow-Neurons_dataframe.p', 'rb')
    shadow = pickle.load(data_file)
    data_file.close()

    y_train = shadow['Labels']
    x_train = shadow['Inputs'].to_numpy()

    y_test = target['Labels']
    x_test = target['Inputs'].to_numpy()


    return x_train, y_train, x_test, y_test

def preprocess_data(inputs, labels):
    #this function is used to process the data into usable format.
    #Let images have the shape (..., 1)
    elements = len(inputs[0])
    inputs=np.array([np.array(xi) for xi in inputs])
    inputs = inputs.reshape(-1, elements, 1)
    #one hot encode labels
    # labels = tf.keras.utils.to_categorical(labels, 2)
    labels = np.array(labels).astype(np.float64)
    return inputs, labels

if __name__ == "__main__":
    models = [('DLA-BiLSTM'),
              ('DLA-LSTM'),
              ('DLA'),
              ('ResNet18-BiLSTM'),
              ('ResNet18-LSTM'),
              ('ResNet18'),
              ('DenseNet121-BiLSTM'),
              ('DenseNet121-LSTM'),
              ('DenseNet121'),
              ('VGG-BiLSTM'),
              ('VGG-LSTM'),
              ('VGG')
              ]

    lstm_models = [('DLA-BiLSTM'),
              ('DLA-LSTM'),
              ('ResNet18-BiLSTM'),
              ('ResNet18-LSTM'),
              ('DenseNet121-BiLSTM'),
              ('DenseNet121-LSTM'),
              ('VGG-BiLSTM'),
              ('VGG-LSTM'),
              ]

    nonlstm_models = [
              ('DLA'),
              ('ResNet18'),
              ('DenseNet121'),
              ('VGG'),
              ]

    models = [
                ('DenseNet121-BiLSTM'),
                ('DenseNet121-LSTM'),
                ('DenseNet121'),
                ('VGG-BiLSTM'),
                ('VGG-LSTM'),
                ('VGG')
              ]

    add_lstm = False
    for method_name in models:
        print(f"Training Attack model for {method_name}")
        models = []
        Accuracy = []
        Precision = []
        Recall = []
        Negative_Recall = []
        F1_Score = []
        Data = []
        scores = []

        if add_lstm:
            x_train, y_train, x_test, y_test = load_lstm_data(method_name)
        else:
            x_train, y_train, x_test, y_test = load_data(method_name)

        x_train, y_train = preprocess_data(x_train, y_train)
        x_test, y_test = preprocess_data(x_test, y_test)
        input_shape = (x_train.shape[1],x_train.shape[2])
        lstm_neurons = int(x_train.shape[1] - 12)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model = build_model(1,lstm_neurons,1,l1=128,l2=64, add_lstm = add_lstm, add_vector=True,add_label=False,add_loss=False)
        opt = Adam(lr = 0.0001)
        model.compile(loss = 'binary_crossentropy', optimizer = opt,metrics = ['accuracy'])
        history = model.fit(x_train, y_train, epochs = 100, validation_data = (x_test, y_test), verbose =1,batch_size=256 ,callbacks=[callback])
        train_predictions = model.predict(x_train)
        train_predictions_labels = []
        for pred in train_predictions:
            if pred > 0.5:
                train_predictions_labels.append(1)
            else:
                train_predictions_labels.append(0)

        test_predictions = model.predict(x_test)
        test_predictions_labels = []
        for pred in test_predictions:
            if pred > 0.5:
                test_predictions_labels.append(1)
            else:
                test_predictions_labels.append(0)

        y_train_label = []
        for pred in y_train:
            if pred > 0.5:
                y_train_label.append(1)
            else:
                y_train_label.append(0)

        y_test_label = []
        for pred in y_test:
            if pred > 0.5:
                y_test_label.append(1)
            else:
                y_test_label.append(0)

        test_accuracy = model.evaluate(x_test, y_test)[1]
        train_accuracy = model.evaluate(x_train, y_train)[1]

        sets = ["train", "test"]
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
            Recall.append(locals()[f'recall_{set_type}'])
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
        d.to_csv(f'{method_name}_attack_models.csv')

        print("train accuracy",train_accuracy)
        print("test accuracy",test_accuracy)
        scores.append(test_accuracy)
