import numpy as np
import tensorflow as tf
import pandas as pd
from DataPrep import DataPreparation
tf.keras.backend.set_floatx('float64')


a=1
class StaticNet(tf.keras.Model):
    def __init__(self,N_hid_layers,size_layers=100,penalty='l1',w=.1,
                 activation='relu'):
        super(StaticNet,self).__init__()
        self.N_layers= N_hid_layers
        self.size_layers = size_layers
        penalty_map = {'l1': tf.keras.regularizers.L1(l1=w),
                       'l2': tf.keras.regularizers.L2(l2=w),
                       'l1l2': tf.keras.regularizers.L1L2(l1=w[0],l2=w[1]),
                       'none': None}
        self.inpLayer= tf.keras.layers.Dense(size_layers,activation=activation,
                                             kernel_regularizer=penalty_map[penalty])
        self.hidd_layers=[tf.keras.layers.Dense(size_layers,activation='relu')
                          for _ in range(self.N_layers)]

    @tf.function
    def call(self,x):
        x= self.inpLayer(x)
        for layer in self.hidd_layers:
            x = layer(x)
        return x


class SeqEncoder(tf.keras.Model):
    def __init__(self, enc_layers,enc_units, batch_size):
        super(SeqEncoder, self).__init__()
        self.enc_units = enc_units
        self.enc_layers= enc_layers
        self.batch_size = batch_size
        self.lstm_layers=[]
        if enc_layers < 1:
            raise Exception("Number of layers must be greater than 1")
        else:
            for i in range(enc_layers):
                self.lstm_layers.append(
                    tf.keras.layers.LSTM(self.enc_units,return_sequences=True,
                                         return_state=True,recurrent_initializer='glorot_uniform',
                                         name='encoder_lstm_layer_{}'.format(i)))

    @tf.function
    def call(self, x, state):
        for layer in self.lstm_layers:
            x, *state = layer(x, initial_state = state)
        return x, state


