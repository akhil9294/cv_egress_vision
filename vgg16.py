#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:08:28 2024

@author: akhil5.gupta
"""




# Step 3: Loading the Base Model
import tensorflow as tf 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers 
from tensorflow.keras import Model 


def model_vgg16(train_generator,validation_generator, epochs, batch_size=None, learning_rate=0.0001, verbose='auto'):
    base_model = VGG16(
        input_shape = (224, 224, 3), # Shape of our images
        include_top = False, # Leave out the last fully connected layer
        weights = 'imagenet')
    
    
    
    for layer in base_model.layers:
        layer.trainable = True
    
    
    # Step 4: Compile and Fit
    
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)
    
    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)
    
    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)
    
    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(1, activation='relu')(x)
    
    model = tf.keras.models.Model(base_model.input, x)
    
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate), loss = 'binary_crossentropy', metrics = ['acc'])
    
    
    vgghist = model.fit(
        train_generator, 
        validation_data=validation_generator, 
        batch_size=batch_size,
        # steps_per_epoch=5, 
        epochs=epochs,
        workers=-1, 
        use_multiprocessing=True,
        verbose=verbose)
    
    return vgghist,model