from keras import initializers, layers
from keras import backend as K
from rnn_Input_layer import rnn_input_layer
from keras.layers import Concatenate, Reshape, Bidirectional, CuDNNLSTM, LSTM
import keras
import tensorflow as tf


class renet_module(keras.Model):
    def __init__(self, X_height, X_width, dim = 3, receptive_filter_size = 2, batch_size = 1, hidden_size = 320):
        super(renet_module,self).__init__()

        self.dim = dim
        self.hidden_size = hidden_size
        self.receptive_filter_size = receptive_filter_size

        self.rnn_input_layer = rnn_input_layer(dim,receptive_filter_size)
        self.lstm = CuDNNLSTM(hidden_size, return_sequences=True)

        self.concatenate = Concatenate(axis = 2)
        self.Reshape = Reshape((int(X_height/self.receptive_filter_size), int(X_width/self.receptive_filter_size), -1))

    def call(self,X):
        print(X.shape)

        _, X_height, X_width, X_channel= X.get_shape()
        vertical_rnn_inputs_fw,vertical_rnn_inputs_rev,horizontal_rnn_inputs_fw,horizontal_rnn_inputs_rev = self.rnn_input_layer(X)
        renet1 = self.lstm(vertical_rnn_inputs_fw)
        renet2 = self.lstm(vertical_rnn_inputs_rev)
        renet3 = self.lstm(horizontal_rnn_inputs_fw)
        renet4 = self.lstm(horizontal_rnn_inputs_rev)
        renet_concat = self.concatenate([renet1, renet2, renet3, renet4])
        renet = self.Reshape(renet_concat)

        return renet

    def compute_output_shape(self,input_shape):
        return (input_shape[0], input_shape[1]/self.receptive_filter_size, input_shape[2]/self.receptive_filter_size, self.hidden_size*4)