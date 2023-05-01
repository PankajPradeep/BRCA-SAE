#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras

class SmallDecoder(keras.Model):
    def __init__(self, input_dim, decode_dim):
        super().__init__()
        self.input_dim = input_dim
        self.decode_dim = decode_dim
        self.decoder_small = keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.input_dim,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(momentum=0.8),
            keras.layers.Dense(self.decode_dim)
    
        ])

    def call(self, inputs):
        return self.decoder_small(inputs)

