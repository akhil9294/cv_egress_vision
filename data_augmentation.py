#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:05:17 2024

@author: akhil5.gupta
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os

    

def f_data_augumentation(train_dir, validation_dir, aug_train_dir, aug_validation_dir, target_size):
      
    aug_train_dir_0 = os.path.join(aug_train_dir,'0')
    aug_train_dir_1 = os.path.join(aug_train_dir,'1')

    aug_validation_dir_0 = os.path.join(aug_validation_dir,'0')
    aug_validation_dir_1 = os.path.join(aug_validation_dir,'1')

    # Step 1: Image Augmentation
    
    # Add our data-augmentation parameters to ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator( rescale = 1.0/255. )
    
    
    # Step 2: Training and Validation Sets
    
    # Flow training images in batches of 20 using train_datagen generator

    
    for i in range(10):
        batches = train_datagen.flow_from_directory(train_dir,save_to_dir = aug_train_dir_1,save_prefix='aug', classes=['1'], batch_size = 100, class_mode = 'binary', target_size = target_size)
        batches.next()
        batches = train_datagen.flow_from_directory(train_dir,save_to_dir = aug_train_dir_0,save_prefix='aug', classes=['0'], batch_size = 100, class_mode = 'binary', target_size = target_size)
        batches.next()
        batches = train_datagen.flow_from_directory(validation_dir,save_to_dir = aug_validation_dir_1,save_prefix='aug', classes=['1'], batch_size = 100, class_mode = 'binary', target_size = target_size)
        batches.next()
        batches = train_datagen.flow_from_directory(validation_dir,save_to_dir = aug_validation_dir_0,save_prefix='aug', classes=['0'], batch_size = 100, class_mode = 'binary', target_size = target_size)
        batches.next()
    
    
    
    train_generator = train_datagen.flow_from_directory(aug_train_dir, batch_size = 100, class_mode = 'binary', target_size = target_size)
    
    
    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory( aug_validation_dir,  batch_size = 100, class_mode = 'binary', target_size = target_size)
    
    
    return train_datagen, test_datagen, train_generator,validation_generator










