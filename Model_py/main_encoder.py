#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.keras as keras

class MainEncoder(keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(MainEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder_inside = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self.input_dim,)),
            keras.layers.BatchNormalization(momentum=0.3),
            keras.layers.Dense(64),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32),
            keras.layers.Dense(self.latent_dim),
            keras.layers.BatchNormalization(momentum=0.2)
        ])

        self.stacked_encoder_inside = keras.models.Sequential([
            self.encoder_inside,
            keras.layers.Dense(32,activation = 'relu'),
            keras.layers.BatchNormalization(momentum=0.9),
            keras.layers.Dense(16,activation = 'relu'),
            keras.layers.BatchNormalization(momentum=0.9),
            keras.layers.Dense(self.latent_dim,activation = 'relu'),
            keras.layers.BatchNormalization(momentum=0.9)
            
        ])

    def call(self, x):
        return self.stacked_encoder_inside(x)

