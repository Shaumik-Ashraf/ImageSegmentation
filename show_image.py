# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 04:38:17 2021

@author: ShaumikAshraf
"""

import os;
import cv2;
import csv;
import json;
import numpy as np;
import pandas as pd;
from PIL import Image, ImageDraw;
from matplotlib import pyplot as plt;

BASE_NAME = "0486052bb";
PRINT_IMAGE = False
PRINT_MASK = True;

Image.MAX_IMAGE_PIXELS = None;


root = os.path.dirname( os.path.abspath(__file__) );
image_path = os.path.join(root, 'data', 'train', BASE_NAME + '.tiff');
json_path = os.path.join(root, 'data', 'train', BASE_NAME + '.json');
anatomy_path = os.path.join(root, 'data', 'train', BASE_NAME + '-anatomical-structure.json')


img = Image.open(image_path);
img_arr = np.array(img); #dtype automatically uint8_t

print(img_arr.shape);

if PRINT_IMAGE:
    plt.figure();
    plt.imshow(img_arr);
    plt.title(BASE_NAME);

with open(anatomy_path) as f:
  j1 = json.load(f)
with open(json_path) as f:
  j2 = json.load(f)

polys = []
for index in range(len(j2)):
    geom = np.array(j2[index]['geometry']['coordinates'])
    polys.append(geom)

h, w, _ = img_arr.shape;
mask_1 = np.zeros((h, w), dtype=np.int32);
for i in range(len(polys)):
    cv2.fillPoly(mask_1, polys[i], i+1)

if PRINT_MASK:
    plt.figure();
    plt.imshow(mask_1);
    plt.title('mask');
