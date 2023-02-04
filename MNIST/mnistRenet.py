# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:32:23 2023

@author: omars
"""
from model import build_model
from tensorflow.keras.optimizers import Adam,SGD
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data():
    data_file = open('mnist_shadow_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('mnist_shadow_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    return data, labels

def prepare_sets(inputs, labels,number_of_classes):
    #this function is used to process the data into usable format.
    #convert inputs to float type and normalize to to range 0,1
    inputs = inputs.astype("float32") / 255.0
    inputs = inputs.reshape(-1,28,28,1)
    #Let images have the shape (..., 1)
    # inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)
    x_train, x_test, y_train, y_test= train_test_split(inputs, labels, stratify=labels, test_size=0.50, random_state=42)


    return x_train, y_train , x_test, y_test


def train(x_train, y_train):
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon = 0.1, decay = 1e-6)

    model = build_model( nClasses=2)
    model.compile(loss='binary_crossentropy',optimizer= opt ,metrics=['accuracy'])

    History=model.fit(x_train,y_train,batch_size=1,epochs=5)

if __name__ == '__main__':
    inputs, labels = load_data()

    num_classes = 10

    x_train, y_train , x_test, y_test = prepare_sets(inputs, labels, num_classes)
    train(x_train, y_train)