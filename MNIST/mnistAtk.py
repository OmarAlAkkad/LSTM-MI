# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:13:41 2023

@author: omars
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D, LSTM, BatchNormalization
import pandas as pd
import xgboost
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score

def load_data():
    data_file = open(f'target_renet_dataframe_mnist.p', 'rb')
    target = pickle.load(data_file)
    data_file.close()

    data_file = open(f'shadow_renet_dataframe_mnist.p', 'rb')
    shadow = pickle.load(data_file)
    data_file.close()

    y_train = shadow['Labels'].to_numpy()
    x_train = shadow['Inputs']

    y_test = target['Labels'].to_numpy()
    x_test = target['Inputs']


    return np.array(x_train), y_train, np.array(x_test), y_test

def preprocess_data(inputs, labels):
    #this function is used to process the data into usable format.
    #Let images have the shape (..., 1)
    total = []
    inputs = np.array(inputs)
    for i in range(len(inputs)):
        total.append(inputs[i])
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, 2)
    labels = np.array(labels)
    total = np.array(total)
    return total.reshape(-1,len(inputs[0]),1), labels

def create_model(input_shape):

    model = Sequential() #initialize model
    model.add(tf.keras.Input(shape=(input_shape)))
    model.add(Flatten()) #flatten the array to input to dense layer
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax')) #output layer with softmax activation function to get predictions vector
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model # return the created model

def xgboost_model(x_train,y_train,x_test,y_test):
    model_xgboost = xgboost.XGBClassifier(learning_rate=0.05,
                                          max_depth=2,
                                          n_estimators=5000,
                                          subsample=0.5,
                                          colsample_bytree=0.25,
                                          eval_metric='auc',
                                          verbosity=1,
                                          use_label_encoder=False)

    eval_set = [(x_test, y_test)]

    model_xgboost.fit(x_train,
                      y_train,
                      early_stopping_rounds=10,
                      eval_set=eval_set,
                      verbose=True)

    evaluation_results = model_xgboost.evals_result()

    y_train_pred = model_xgboost.predict_proba(x_train)[:,1]
    y_test_pred = model_xgboost.predict_proba(x_test)[:,1]

    print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                        roc_auc_score(y_test, y_test_pred)))


if __name__ == "__main__":
    models = []
    Accuracy = []
    Precision = []
    Recall = []
    Negative_Recall = []
    F1_Score = []
    Data = []
    scores = []

    x_train, y_train, x_test, y_test = load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # xgboost_model(x_train,y_train,x_test,y_test)

    input_shape = (x_train.shape[1],x_train.shape[2])
    model = create_model(input_shape)
    history = model.fit(x_train, y_train, epochs = 1000, validation_data = (x_test, y_test), verbose =1,batch_size= 150)

    train_predictions = model.predict(x_train)
    train_predictions_labels = []
    for pred in train_predictions:
        if pred[1] > pred[0]:
            train_predictions_labels.append(1)
        else:
            train_predictions_labels.append(0)

    test_predictions = model.predict(x_test)
    test_predictions_labels = []
    for pred in test_predictions:
        if pred[1] > pred[0]:
            test_predictions_labels.append(1)
        else:
            test_predictions_labels.append(0)

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
    d.to_csv(f'mnist_attack_models.csv')

    print("train accuracy",train_accuracy)
    print("test accuracy",test_accuracy)
    scores.append(test_accuracy)
