'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)

'''
from keras.models import *
from keras.layers import *
#from Input_layer import vertical_layer
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Lambda, Reshape, Bidirectional, CuDNNLSTM, Dense, Flatten
from renet_layer import renet_module
import keras

class build_model(keras.Model):

    def __init__(self, nClasses , input_height, input_width):
        super(build_model,self).__init__()

        self.renet_module = renet_module(X_height=input_height, X_width=input_width, dim=3,receptive_filter_size=4, batch_size=1, hidden_size=320)
        self.conv = Conv2D(1, kernel_size=(1, 1))
        self.upsample = convolutional.UpSampling2D(size=(4, 4), data_format=None)
        self.flatten = Flatten()
        self.dense = Dense(50, activation = 'relu')
        self.dense1 = Dense(50, activation = 'relu')
        self.predict = Dense(nClasses,activation = 'softmax')

    def call(self, inputs):
        renet = self.renet_module(inputs)
        print('renet',renet.shape)
        conv = self.conv(renet)
        upsample = self.upsample(conv)
        print('upsample', upsample.shape)
        flattened = self.flatten(upsample)
        dense1 = self.dense(flattened)
        dense2 = self.dense1(dense1)
        outputs = self.predict(dense2)
        print(outputs.eval)

        return outputs