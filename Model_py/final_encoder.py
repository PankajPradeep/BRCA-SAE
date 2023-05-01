#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow.keras as keras

class FinalEncoder(keras.Model):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder_conc = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(32,activation = 'relu'),
            keras.layers.Dense(16,activation = 'relu'),
            keras.layers.BatchNormalization(momentum=0.2),
            keras.layers.Dense(latent_dim),
            keras.layers.BatchNormalization(momentum=0.2)
        ])

        # Stack the encoder layers
        self.stacked_encoder_conc = keras.models.Sequential([
            self.encoder_conc,
            keras.layers.Dense(32),
            keras.layers.BatchNormalization(momentum=0.9),     
            keras.layers.Dense(16),
            keras.layers.BatchNormalization(momentum=0.8),
            keras.layers.Dense(latent_dim),
            keras.layers.BatchNormalization(momentum=0.7)
        ])

    def call(self, inputs):
        return self.stacked_encoder_conc(inputs)

