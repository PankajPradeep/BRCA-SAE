#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow.keras as keras

class BigDecoder(keras.Model):
    def __init__(self, input_dim, decode_dim):
        super().__init__()
        self.input_dim = input_dim
        self.decode_dim = decode_dim
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(16,activation = 'relu'),
            keras.layers.Dense(32,activation = 'relu'),
            keras.layers.Dense(64,activation = 'relu'),
            keras.layers.Dense(128,activation = 'relu'),
            keras.layers.Dense(256,activation = 'relu'),
            #keras.layers.Dense(512,activation = 'relu'),
            keras.layers.Dense(decode_dim)
        ])

    def call(self, inputs):
        return self.decoder(inputs)

