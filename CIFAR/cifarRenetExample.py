# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 19:23:59 2023

@author: omars
"""
import pickle
import math
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import build_model
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

def create_dataset():
    #load mnist dataset from keras.
    train, test = tf.keras.datasets.cifar10.load_data()

    x = train[0]
    y = train[1]

    x_ = test[0]
    y_ = test[1]

    # inputs = []
    # for i in range(len(y)):
    #     inputs.append(x[i])

    # for i in range(len(y_)):
    #     inputs.append(x_[i])

    # inputs = np.array(inputs)

    # labels = []
    # for i in range(len(y)):
    #     labels.append(y[i])

    # for i in range(len(y_)):
    #     labels.append(y_[i])

    # labels = np.array(labels)

    return x, y, x_, y_

def prepare_sets(inputs, labels,number_of_classes):
    #this function is used to process the data into usable format.
    #convert inputs to float type and normalize to to range 0,1
    #inputs = inputs.reshape(-1,1)
    #scaler = StandardScaler()
    #inputs = scaler.fit_transform(inputs, labels)
    # inputs = inputs.reshape(-1,32,32,3)
    inputs = inputs.astype('float32') /255.0
    inputs = inputs.reshape(-1,32,32,3)
    #Let images have the shape (..., 1)
    #inputs = np.expand_dims(inputs, -1)
    #one hot encode labels
    labels = tf.keras.utils.to_categorical(labels, number_of_classes)

    # inputs, _, labels , y__ = train_test_split(inputs, labels, stratify=labels, test_size=0.01, random_state=42)

    return np.array(inputs), np.array(labels)

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

def augment_images(data, labels):
    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
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
    labels = np.array(labels).reshape(-1,10)

    return images,labels


if __name__ == "__main__":
    number = 10
    x_train, y_train, x_test, y_test = create_dataset()

    #x_train, y_train = augment_data(x_train, y_train)

    x_train, y_train = prepare_sets(x_train, y_train, number)
    x_test, y_test = prepare_sets(x_test, y_test, number)

    y_train_label = []
    for pred in y_train:
        y_train_label.append(np.argmax(pred, axis=0))

    opt = Adam(learning_rate = 0.0001)
    model = build_model(10, 32, 32, 30)
    model.compile(loss='categorical_crossentropy',optimizer= opt ,metrics=['accuracy'])
    checkpoint_path = "training_example/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history=model.fit(x_train,y_train,batch_size=30,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])

    model.fit(x_train, y_train, batch_size=30, validation_data = (x_test, y_test), epochs=100, callbacks=[cp_callback])

    #history=model.fit(x_train,y_train,batch_size=5,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])

    train_predictions = np.empty(0, dtype="float32")
    test_predictions = np.empty(0, dtype= 'float32')

    for x in range(len(x_train)):
        train_predictions = np.append(train_predictions, np.array(K.eval(model.predict(np.expand_dims(x_train[x], axis=0)))))

    for x in range(len(x_test)):
        test_predictions = np.append(test_predictions, np.array(K.eval(model.predict(np.expand_dims(x_test[x], axis=0)))))

    train_predictions = train_predictions.reshape(-1,num_classes)
    test_predictions = test_predictions.reshape(-1,num_classes)

    train_predictions_labels = []
    for pred in train_predictions:
        train_predictions_labels.append(np.argmax(pred, axis=0))

    test_predictions_labels = []
    for pred in test_predictions:
        test_predictions_labels.append(np.argmax(pred, axis=0))

    y_train_label = []
    for pred in y_train:
        y_train_label.append(np.argmax(pred, axis=0))

    y_test_label = []
    for pred in y_test:
        y_test_label.append(np.argmax(pred, axis=0))

    train_losses = tf.keras.backend.categorical_crossentropy(y_train, train_predictions).numpy()
    test_losses = tf.keras.backend.categorical_crossentropy(y_test, test_predictions).numpy()

    train_predictions_list = train_predictions.tolist()
    test_predictions_list = test_predictions.tolist()

    inputs = []
    all_labels = []
    attention_weights = []

    for i in range(len(train_predictions)):
        # if train_predictions_labels[i] == y_train_label[i]:
            train_predictions_list[i].append(train_losses[i])
            train_predictions_list[i].append(y_train_label[i])
            inputs.append(train_predictions_list[i])
            all_labels.append(1)
    for i in range(len(test_predictions)):
        # if test_predictions_labels[i] == y_test_label[i]:
            test_predictions_list[i].append(test_losses[i])
            test_predictions_list[i].append(y_test_label[i])
            inputs.append(test_predictions_list[i])
            all_labels.append(0)

    temp = list(zip(inputs, all_labels))
    random.shuffle(temp)
    inputs, all_labels = zip(*temp)
    inputs, all_labels = list(inputs), list(all_labels)
    d = {
        'Inputs': inputs,
        'Labels': all_labels
        }
    locals()['shadow_renet_dataframe_cifar'] = pd.DataFrame(data=d)

    pickle.dump(locals()['shadow_renet_dataframe_cifar'], open('shadow_renet_dataframe_cifar.p', 'wb'))

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
        locals()[f'precision_{set_type}'] = precision_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='macro')
        locals()[f'recall_{set_type}'] = recall_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'],average='macro')

        locals()[f'f1_{set_type}'] = f1_score(locals()[f'y_{set_type}_label'], locals()[f'{set_type}_predictions_labels'], average='macro')
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
    d.to_csv(f'shadow_renet_cifar.csv')

