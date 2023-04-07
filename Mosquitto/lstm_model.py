# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:04:23 2023

@author: omars
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:14:03 2023

@author: omars
"""
from keras.models import *
from keras.layers import *
#from Input_layer import vertical_layer
import numpy as np
import tensorflow as tf
from keras.layers import Dropout, Conv2D, Lambda, Reshape, Bidirectional, CuDNNLSTM, Dense, Flatten,LSTM
from renet_layer import renet_module
import keras

class build_model(keras.Model):

    def __init__(self, num_classes):
        super(build_model,self).__init__()

        self.lstm1 = LSTM(1024, activation="tanh",return_sequences=True)
        self.lstm2 = LSTM(512, activation = 'tanh')
        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.dense1 = Dense(512, activation="relu")
        self.dense2 = Dense(128, activation="relu")
        self.dense3 = Dense(64, activation="relu")
        self.outputs = Dense(num_classes, activation = 'softmax')

    def call(self, inputs):
        lstm1 = self.lstm1(inputs)
        lstm2 = self.lstm2(lstm1)
        lstm2 = self.dropout(lstm2)
        dense1 = self.dense1(lstm2)
        dense2 = self.dense2(dense1)
        dense2 = self.dropout(dense2)
        dense3 = self.dense3(dense2)
        outputs = self.outputs(dense3)

        return outputs

    def lstm(self, inputs):
        lstm = self.lstm1(inputs)

        return lstm
