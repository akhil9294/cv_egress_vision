#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:13:26 2024

@author: akhil5.gupta
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def f_display_image_data(train_0, train_1):
    
    
    train_0_fnames = os.listdir( train_0 )
    train_1_fnames = os.listdir( train_1 )
    
    next_0_pix = [os.path.join(train_0, fname) for fname in train_0_fnames[ 0:6] ]
    next_1_pix = [os.path.join(train_1, fname) for fname in train_1_fnames[ 0:4] ]
    
    nrows = 4
    ncols = 4
    
    
    # image = Image.open('dataset/0/15218948102_008e3d8efe_b.jpg')
    # new_image = image.resize((500, 500))
    # new_image.save('myimage_500.jpg')
    
    # for i, img_path in enumerate(next_0_pix):
    #     image = Image.open(img_path)
    #     a = image.resize((224,224))
    #     a.save(train_0+'/'+str(i)+'.jpg')
    
    # for i, img_path in enumerate(next_1_pix):
    #     image = Image.open(img_path)
    #     a = image.resize((224,224))
    #     a.save(train_1+'/'+str(i)+'.jpg')
    
    train_0_fnames = os.listdir( train_0 )
    train_1_fnames = os.listdir( train_1 )
    next_0_pix = [os.path.join(train_0, fname) for fname in train_0_fnames[ 0:7] ]
    next_1_pix = [os.path.join(train_1, fname) for fname in train_1_fnames[ 0:5] ]
    # print(next_0_pix)
    # print(next_1_pix)
    # next_0_pix.pop(0)
    # next_0_pix
    # next_1_pix.pop(0)
    # next_1_pix
    
    for i, img_path in enumerate(next_0_pix+next_1_pix):
      # Set up subplot; subplot indices start at 1
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)
    
      img = mpimg.imread(img_path)
      plt.imshow(img)
    
    plt.show()
