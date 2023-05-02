import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate

class build_model(keras.Model):

    def __init__(self, nClasses,lstm_neurons, l1=128, l2=64):
        super(build_model,self).__init__()

        self.flatten = Flatten()
        self.BatchNormalization = BatchNormalization()
        self.dropout = Dropout(0.2)
        self.l1 = Dense(l1, activation = 'relu')
        self.l2 = Dense(l2, activation = 'relu')
        self.l1_e = Dense(l1, activation = 'relu')
        self.l2_e = Dense(l2, activation = 'relu')
        self.denselstm = Dense(lstm_neurons, activation='relu')
        self.dense320 = Dense(320, activation = 'relu')
        self.dense256 = Dense(256, activation = 'relu')
        self.dense320_e = Dense(320, activation = 'relu')
        self.dense256_e = Dense(256, activation = 'relu')
        self.dense256_e1 = Dense(256, activation = 'relu')
        self.dense128 = Dense(128, activation = 'relu')
        self.dense64 = Dense(64, activation = 'relu')
        self.outputlayer = Dense(nClasses,activation = 'softmax')

    def call(self, inputs):
        vectors_slice = inputs[:, :10]
        loss_slice = inputs[:, 10:11]
        label_slice = inputs[:, 11:12]
        lstm_slice = inputs[:, 12:]


        lstm_out = self.process_lstm(lstm_slice)
        loss_out = self.loss_output(loss_slice)
        vector_out = self.vector_output(vectors_slice)
        label_out = self.label_output(label_slice)

        concat = concatenate([lstm_out, vector_out, loss_out, label_out], axis = 1)

        encoder = self.encoder(inputs)

        return encoder

    def process_lstm(self, inputs):
        dropout = self.dropout(inputs)
        dense1 = self.denselstm(dropout)
        dense2 = self.l1(dense1)
        dense3 = self.l2(dense2)

        return dense3

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
        inputs_normalized = self.BatchNormalization(inputs_flattened)
        dropout_e = self.dropout(inputs_normalized)
        dropout_e_normalized = self.BatchNormalization(dropout_e)
        dense_e = self.dense320_e(dropout_e_normalized)
        dense1_e = self.dense256_e(dense_e)
        dropout1_e = self.dropout(dense1_e)
        dropout1_e_normalized = self.BatchNormalization(dropout1_e)
        dense2_e = self.dense256_e1(dropout1_e_normalized)
        dense3_e = self.l1_e(dense2_e)
        dropout2_e = self.dropout(dense3_e)
        dropout2_e_normalized = self.BatchNormalization(dropout2_e)
        dense4_e = self.l2_e(dropout2_e_normalized)
        outputs = self.outputlayer(dense4_e)
        return outputs
