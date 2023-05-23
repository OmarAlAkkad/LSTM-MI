import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate

class build_model(keras.Model):

    def __init__(self, nClasses,lstm_neurons, percentage_neurons,l1=128, l2=64, add_lstm = True):
        super(build_model,self).__init__()
        self.add_lstm = add_lstm
        self.neurons = lstm_neurons
        self.percent = percentage_neurons
        self.flatten = Flatten()
        self.batchnormalization = BatchNormalization()
        self.dropout = Dropout(0.2)
        self.l1 = Dense(l1, activation = 'relu')
        self.l2 = Dense(l2, activation = 'relu')
        self.l1_e = Dense(l1, activation = 'relu')
        self.l2_e = Dense(l2, activation = 'relu')
        self.denselstm = Dense(int(lstm_neurons*percentage_neurons), activation='relu')
        self.dense320 = Dense(320, activation = 'relu')
        self.dense256 = Dense(256, activation = 'relu')
        self.dense320_e = Dense(320, activation = 'relu')
        self.dense256_e = Dense(256, activation = 'relu')
        self.dense256_e1 = Dense(256, activation = 'relu')
        self.dense128 = Dense(128, activation = 'relu')
        self.dense64 = Dense(64, activation = 'relu')
        self.outputlayer = Dense(nClasses,activation = 'sigmoid')

    def call(self, inputs):
        vectors_slice = inputs[:, :10]
        loss_slice = inputs[:, 10:11]
        label_slice = inputs[:, 11:12]
        loss_out = self.loss_output(loss_slice)
        vector_out = self.vector_output(vectors_slice)
        label_out = self.label_output(label_slice)

        concat = concatenate([vector_out, loss_out, label_out], axis = 1)

        if self.add_lstm:
            lstm_slice = inputs[:, 12:int(self.neurons*self.percent)+12]
            lstm_out = self.process_lstm(lstm_slice)
            concat = concatenate([vector_out, loss_out, label_out, lstm_out], axis = 1)

        encoder = self.encoder(concat)

        return encoder

    def process_lstm(self, inputs):
        dropout = self.dropout(inputs)
        dense1 = self.l1(dropout)
        dense2 = self.l2(dense1)

        return dense2

    def loss_output(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense128(dropout)
        dense1 = self.dense64(dense)

        return dense1

    def vector_output(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense128(dropout)
        dense1 = self.dense64(dense)

        return dense1

    def label_output(self, inputs):
        dropout = self.dropout(inputs)
        dense = self.dense128(dropout)
        dense1 = self.dense64(dense)

        return dense1

    def encoder(self, inputs_e):
        inputs_flattened = self.flatten(inputs_e)
        dropout_e = self.dropout(inputs_flattened)
        dense_e = self.dense320_e(dropout_e)
        dense1_e = self.dense256_e(dense_e)
        dropout1_e = self.dropout(dense1_e)
        dense2_e = self.dense256_e1(dropout1_e)
        dense3_e = self.l1_e(dense2_e)
        dropout2_e = self.dropout(dense3_e)
        dense4_e = self.l2_e(dropout2_e)
        outputs = self.outputlayer(dense4_e)
        return outputs

