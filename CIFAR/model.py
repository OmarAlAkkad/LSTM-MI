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

    def __init__(self, nClasses , input_height, input_width, batch_size):
        super(build_model,self).__init__()

        self.renet_module = renet_module(X_height=input_height, X_width=input_width, dim=3,receptive_filter_size=4, batch_size=batch_size, hidden_size=320)
        self.conv = Conv2D(1, kernel_size=(1, 1))
        self.upsample = convolutional.UpSampling2D(size=(4, 4), data_format=None)
        self.flatten = Flatten()
        self.dense = Dense(4096, activation = 'relu')
        # self.dense1 = Dense(128, activation = 'relu')
        self.outputlayer = Dense(nClasses,activation = 'softmax')

    def call(self, inputs):
        renet = self.renet_module(inputs)
        #conv = self.conv(renet)
        #upsample = self.upsample(conv)
        flattened = self.flatten(renet)
        dense1 = self.dense(flattened)
        # dense2 = self.dense1(dense1)
        outputs = self.outputlayer(dense1)

        return outputs
