#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:33:49 2024

@author: akhil5.gupta
"""



import os 


from vgg16 import model_vgg16
from inceptionV3 import f_model_inceptionV3
from data_augmentation import f_data_augumentation
from display_image_data import f_display_image_data
from resize_image import f_resize_raw_image


base_dir = 'dataset/'

raw_train_dir = os.path.join(base_dir, 'raw_train')
raw_validation_dir = os.path.join(base_dir, 'raw_validation')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

aug_train_dir = os.path.join(base_dir,'aug_train')
aug_validation_dir = os.path.join(base_dir,'aug_validation')


target_size_raw = (224,224)

## Image resize
f_resize_raw_image(raw_train_dir+'/0',train_dir+'/0',target_size_raw)
f_resize_raw_image(raw_train_dir+'/1',train_dir+'/1',target_size_raw)
f_resize_raw_image(raw_validation_dir+'/0',validation_dir+'/0',target_size_raw)
f_resize_raw_image(raw_validation_dir+'/1',validation_dir+'/1',target_size_raw)

## Image display.
#f_display_image_data(train_0,train_1)


## Augmentation of images and staore in aug directory.
target_size = (224,224)
train_datagen, test_datagen, train_generator,validation_generator =  f_data_augumentation(train_dir, validation_dir, aug_train_dir, aug_validation_dir, target_size)
## Train VGG16 model on augumented data.
vgghist,model_vgg16 = model_vgg16(train_generator,validation_generator)
hist, model = vgghist,model_vgg16

## Augmentation of images and staore in aug directory.
target_size = (150,150)
train_datagen, test_datagen, train_generator,validation_generator =  f_data_augumentation(target_size)
## Train InceptionV3 model on augumented data.
inceptionV3history, model_inceptionV3 = f_model_inceptionV3(train_generator,validation_generator)

#os.unlink(aug_train_dir_0)

hist, model = inceptionV3history, model_inceptionV3 



#Result validation on validation dataset.

validation_generator0 = test_datagen.flow_from_directory( validation_dir, classes = ['0'], batch_size = 20, class_mode = 'binary', target_size = target_size)
yhat0 = model.predict(validation_generator0)
yhat0

validation_generator1 = test_datagen.flow_from_directory( validation_dir, classes = ['1'], batch_size = 20, class_mode = 'binary', target_size = target_size)
yhat1 = model.predict(validation_generator1)
yhat1

import pandas as pd
df = pd.DataFrame(yhat0, columns=['predict'])
df['actual'] = '0'

df1 = pd.DataFrame(yhat1, columns=['predict'])
df1['actual'] = '1'

df_final = pd.concat([df,df1])
df_final





import seaborn as sns

#sns.histplot(yhat)
sns.scatterplot(data=df_final, x='actual', y='predict', hue='actual')











