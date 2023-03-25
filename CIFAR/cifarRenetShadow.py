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
from sklearn.metrics import f1_score,recall_score,precision_score, confusion_matrix
import random
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2


def load_data():
    data_file = open('cifar_shadow_data.p', 'rb')
    data = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_shadow_labels.p', 'rb')
    labels = pickle.load(data_file)
    data_file.close()

    x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.5, random_state=42)

    y_train = tf.keras.utils.to_categorical(y_train, 10)

    return x_train, x_test, y_train, y_test

def load_augmented_data():
    data_file = open('cifar_shadow_x_train.p', 'rb')
    x_train = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_shadow_x_test.p', 'rb')
    x_test = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_shadow_y_train.p', 'rb')
    y_train = pickle.load(data_file)
    data_file.close()

    data_file = open('cifar_shadow_y_test.p', 'rb')
    y_test = pickle.load(data_file)
    data_file.close()

    x_test = x_test.astype('float32') /255.0
    x_test = x_test.reshape(-1,32,32,3)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test

def load_split_data():
    data_file = open('shadow_train_inputs.p', 'rb')
    x_train = pickle.load(data_file)
    data_file.close()

    data_file = open('shadow_test_inputs.p', 'rb')
    x_test = pickle.load(data_file)
    data_file.close()

    data_file = open('shadow_train_labels.p', 'rb')
    y_train = pickle.load(data_file)
    data_file.close()

    data_file = open('shadow_test_labels.p', 'rb')
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

    opt = Adam(learning_rate = 0.0001)

    model = build_model(10, 32, 32, 25)
    model.compile(loss='categorical_crossentropy',optimizer= opt ,metrics=['accuracy'])

    checkpoint_path = "training_shadow/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1)

    # datagen = ImageDataGenerator(
    # featurewise_center=True,
    # rescale=1.0/255.0,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,)
    # # compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied)
    # datagen.fit(x_train)

    # epochs = 100
    # for e in range(epochs):
    #     print('Epoch', e)
    #     batches = 0
    #     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=30):
    #         history = model.fit(x_batch, y_batch, callbacks=[cp_callback])
    #         batches += 1
    #         if batches >= len(x_train) / 30:
    #             # we need to break the loop by hand because
    #             # the generator loops indefinitely
    #             break

    history=model.fit(x_train,y_train, batch_size=25 ,epochs=100,validation_data = (x_test, y_test), callbacks=[cp_callback])

    # save weights to disk

    # redefine the model with batch size 1, and reload the weight

    model = build_model(10, 32, 32, 1)

    model.load_weights(checkpoint_path)


    print('Train loss:', history.history['loss'])
    print('Train accuracy : ', history.history['accuracy'])
    print('Test loss:', history.history['val_loss'])
    print('Test accuracy : ', history.history['val_accuracy'])
    error_rate = round(1 - history.history['val_accuracy'][0], 3)
    print('error rate of :', error_rate)

    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]

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

    plot_data(history)
