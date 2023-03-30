# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:26:54 2023

@author: omars
"""
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_dataset():
    #load mnist dataset from keras.
    train, test = tf.keras.datasets.cifar10.load_data()
    train_inputs = train[0]
    train_labels = train[1]
    target_x_train, x_test, target_y_train, y_test = train_test_split(train_inputs, train_labels, test_size = 0.8, stratify=train_labels,random_state=42)
    shadow_x_train, x_test, shadow_y_train, y_test = train_test_split(x_test, y_test, test_size = 0.75, stratify=y_test, random_state=42)
    testing_images, x_test, testing_labels, y_test = train_test_split(x_test, y_test, test_size = 2/3, stratify=y_test, random_state=42)
    testing_images1, testing_images2, testing_labels1, testing_labels2 = train_test_split(x_test, y_test, test_size = 1/2, stratify=y_test, random_state=42)

    pickle.dump(testing_images, open('testing_images.p', 'wb'))
    pickle.dump(testing_labels, open('testing_labels.p', 'wb'))
    pickle.dump(testing_images1, open('testing_images1.p', 'wb'))
    pickle.dump(testing_labels1, open('testing_labels1.p', 'wb'))
    pickle.dump(testing_images2, open('testing_images2.p', 'wb'))
    pickle.dump(testing_labels2, open('testing_labels2.p', 'wb'))

    test_inputs = test[0]
    test_labels = test[1]

    return target_x_train, target_y_train, shadow_x_train, shadow_y_train, test_inputs, test_labels


if __name__ == "__main__":
    target_x_train, target_y_train, shadow_x_train, shadow_y_train, test_inputs, test_labels = create_dataset()

    pickle.dump(target_x_train, open('target_train_inputs.p', 'wb'))
    pickle.dump(shadow_x_train, open('shadow_train_inputs.p', 'wb'))
    pickle.dump(target_y_train, open('target_train_labels.p', 'wb'))
    pickle.dump(shadow_y_train, open('shadow_train_labels.p', 'wb'))
    pickle.dump(test_inputs, open('target_test_inputs.p', 'wb'))
    pickle.dump(test_inputs, open('shadow_test_inputs.p', 'wb'))
    pickle.dump(test_labels, open('target_test_labels.p', 'wb'))
    pickle.dump(test_labels, open('shadow_test_labels.p', 'wb'))
















