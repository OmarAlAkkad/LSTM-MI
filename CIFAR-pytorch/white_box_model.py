# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 01:29:04 2023

@author: omars
"""
import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import concatenate

class build_model(keras.Model):

    def __init__(self, nClasses, l1=128, l2=64):
        super(build_model,self).__init__()

        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.l1 = Dense(l1, activation = 'relu')
        self.l2 = Dense(l2, activation = 'relu')
        self.dense320 = Dense(320, activation = 'relu')
        self.dense256 = Dense(256, activation = 'relu')
        self.dense128 = Dense(128, activation = 'relu')
        self.dense64 = Dense(64, activation = 'relu')
        self.outputlayer = Dense(nClasses,activation = 'softmax')

    def call(self, inputs):
        print(inputs)
        lstm = self.lstm(self.lsmts)
        loss = self.loss(self.loss)
        vector = self.vector(self.vector)
        label = self.label(self.label)
        concat = concatenate([lstm,vector,loss,label])
        encoder = self.encoder(concat)

        return encoder

    def lstm(self, inputs):
        dropout = self.dropout(inputs)
        dense1 = self.dense128(dropout)
        dense2 = self.denselstm(dense1)
        dense3 = self.densel1(dense2)
        dense4 = self.densel2(dense3)

        return dense4

    def loss(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense128(dropout)
        dense1 = self.dense64(dense)

        return dense1

    def vector(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense128(dropout)
        dense1 = self.dense64(dense)

        return dense1

    def label(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense128(dropout)
        dense1 = self.dense64(dense)

        return dense1

    def encoder(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense320(dropout)
        dense1 = self.dense256(dense)
        dropout1 = self.dropout(dense1)
        dense2 = self.dense256(dropout1)
        dense3 = self.densel1(dense2)
        dropout2 = self.dropout(dense3)
        dense4 = self.densel2(dropout2)
        outputs = self.outputlayer(dense4)

        return outputs
