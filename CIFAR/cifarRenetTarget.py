"""
Created on Sun Jan 29 17:32:23 2023

@author: omars
"""
from model import build_model
from tensorflow.keras.optimizers import Adam, SGD
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix, accuracy_score
import random
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2


def load_data():
    data_file = open('cifar_target_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_target_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.5, random_state=42)

    y_train = tf.keras.utils.to_categorical(y_train, 10)

    return x_train, x_test, y_train, y_test

def load_augmented_data():
    data_file = open('cifar_target_x_train.p', 'rb')
    x_train = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_target_x_test.p', 'rb')
    x_test = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_target_y_train.p', 'rb')
    y_train = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_target_y_test.p', 'rb')
    y_test = pickle.load(data_file)
    data_file.close()

    x_test = x_test.astype('float32') /255.0
    x_test = x_test.reshape(-1,32,32,3)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test

def load_split_data():
    data_file = open('target_train_inputs.p', 'rb')
    x_train = pickle.load(data_file)
    data_file.close()

    data_file = open('target_test_inputs.p', 'rb')
    x_test = pickle.load(data_file)
    data_file.close()

    data_file = open('target_train_labels.p', 'rb')
    y_train = pickle.load(data_file)
    data_file.close()

    data_file = open('target_test_labels.p', 'rb')
    y_test = pickle.load(data_file)
    data_file.close()

    return x_train, x_test, y_train, y_test

def prepare_sets(inputs, labels,number_of_classes):
    #this function is used to process the data into usable format.
    #convert inputs to float type and normalize to to range 0,1
    #inputs = inputs.reshape(-1,1)
    #scaler = StandardScaler()
    #inputs = scaler.fit_transform(inputs, labels)
    #inputs = inputs.reshape(-1,32,32,3)
    inputs = inputs.astype('float32') /255.0
    inputs = inputs.reshape(-1,32,32,3)
    #Let images have the shape (..., 1)
    #inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)

    return np.array(inputs), np.array(labels)


def plot_data(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def augment_data(inputs, labels):
    flipped = []
    for image in inputs:
        chance = random.randint(0,3)
        if chance == 0:
            flipped.append(cv2.flip(image,1))
        elif chance == 1:
            flipped.append(cv2.flip(image,0))
        else:
            flipped.append(image)

    right = np.float32([[1, 0, 2], [0, 1, 0]])
    left = np.float32([[1, 0, -2], [0, 1, 0]])
    up = np.float32([[1, 0, 0], [0, 1, -2]])
    down = np.float32([[1, 0, 0], [0, 1, 2]])

    shifted = []
    for image in flipped:
        chance = random.randint(0,3)
        if chance == 0:
            shifted.append(cv2.warpAffine(image, right, (image.shape[1], image.shape[0])))
        elif chance == 1:
            shifted.append(cv2.warpAffine(image, left, (image.shape[1], image.shape[0])))
        else:
            shifted.append(image)

    shifted2 = []
    for image in shifted:
        chance = random.randint(0,3)
        if chance == 0:
            shifted2.append(cv2.warpAffine(image, up, (image.shape[1], image.shape[0])))
        elif chance == 1:
            shifted2.append(cv2.warpAffine(image, down, (image.shape[1], image.shape[0])))
        else:
            shifted2.append(image)

    new_train = np.append(shifted2,inputs).reshape(-1,32,32,3)
    new_labels = np.append(labels, labels).reshape(-1,1)

    return np.array(new_train), np.array(new_labels)

def augment_images(data, labels, number_of_classes):
    datagen = ImageDataGenerator(
    featurewise_center=True,
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(data)
    # fits the model on batches with real-time data augmentation:
    aug_images = datagen.flow((data,labels), batch_size=1)

    images = []
    labels = []
    for image in aug_images:
        if len(images) == 100000:
            break
        images.append(image[0])
        labels.append(image[1])

    images = np.array(images).reshape(-1,32,32,3)
    labels = np.array(labels).reshape(-1,1)

    labels = tf.keras.utils.to_categorical(labels, number_of_classes)

    return images,labels

if __name__ == '__main__':
    # x_train, x_test, y_train, y_test = load_data()
    x_train, x_test, y_train, y_test = load_split_data()
    # x_train, x_test, y_train, y_test = load_augmented_data()

    num_classes = 10

    # x_train, y_train = augment_images(x_train, y_train, num_classes)
    x_train, y_train = prepare_sets(x_train, y_train, num_classes)
    x_test, y_test = prepare_sets(x_test, y_test, num_classes)

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon = 0.1)

    model = build_model(10, 32, 32, 5)
    model.compile(loss='categorical_crossentropy',optimizer= opt ,metrics=['accuracy'])

    checkpoint_path = "training_target1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

    history=model.fit(x_train,y_train, batch_size=5 ,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])

    print('Train loss:', history.history['loss'])
    print('Train accuracy : ', history.history['accuracy'])
    print('Test loss:', history.history['val_loss'])
    print('Test accuracy : ', history.history['val_accuracy'])
    error_rate = round(1 - history.history['val_accuracy'][0], 3)
    print('error rate of :', error_rate)

    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]

    plot_data(history)
