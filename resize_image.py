#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:28:30 2024

@author: akhil5.gupta
"""

import os
from PIL import Image


def f_resize_raw_image(src_dir, tgt_dir, target_size):
    
    ls_src_dir_files = [ os.path.join(src_dir, i) for i in os.listdir(src_dir)]
    ls_src_dir_files.remove(src_dir+'/.DS_Store')
    for i, img_path in enumerate(ls_src_dir_files):
        print(img_path)
        print(f'[+] Processing file {img_path}...')
        image = Image.open(img_path)
        image = image.convert('RGB') #convert png to jpg
        a = image.resize(target_size)
        a.save(tgt_dir+'/'+str(i)+'.jpg')
