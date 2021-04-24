# -*- coding: utf-8 -*-
# Pytorch dataset implementation for HuBMAP 


import torch;
import os;
import glob;
import numpy as np;
import pandas as pd;
from torch.utils.data import Dataset;
from scipy.sparse import lil_matrix;
from scipy.sparse import csr_matrix;
from PIL import Image;
from matplotlib import pyplot as plt;


Image.MAX_IMAGE_PIXELS = None;

class KidneyDataset(Dataset):
    """
    Pytorch Dataset class, has the following per sample:
        (image, metadata, mask)
        
        where, image: H x W x 3 array (whatever PIL.Image.load() outputs)
               metadata: dict with keys:
                   width_pixels
                   height_pixels	
                   anatomical_structures_segmention_file	
                   glomerulus_segmentation_file	
                   patient_number	
                   race	
                   ethnicity	
                   sex	
                   age	
                   weight_kilograms	
                   height_centimeters	
                   bmi_kg/m^2	
                   laterality	
                   percent_cortex	
                   percent_medulla
               mask: H x W sparse matrix where 1 is a FTU pixel
               
        Files needed:
            data/train.csv
            data/HuBMAP-20-dataset_information.csv
            data/<img_dir>/*.tiff exist
    """
    
    # ======================= required methods =============================
    def __init__(self, img_dir, train = True, transform = None):
        """
        Params:
            img_dir (string) - folder with images, needs at least one .tiff image
            train (boolean) - default: True
            transform - pytorch transforms or transforms.Compose()
        """
        # save image directory
        self.img_dir = img_dir;
        
        # only collect files available in image_dir
        self.files = [];
        for file in os.listdir(img_dir):
            if( file.find('.tiff') > 0 ):
                self.files.append(file);
        
        # collect metadata of files in img_dir only
        df = self.read_csv('HuBMAP-20-dataset_information.csv');
        self.metadata = dict();
        for file in self.files:
            file_data = df.query("image_file==@file");
            self.metadata[file] = dict();
            for column in file_data.columns:
                self.metadata[file][column] = file_data[column].values[0];
        #print(self.metadata);
        
        # if train, assume data/train.csv and parse run length encoding
        if train:
            df = self.read_csv('train.csv');
            df['id'] = df['id'].apply(lambda s: s + '.tiff');
            self.labels = dict();
            
            for file in self.files:
                rle = df[ df['id']==file ]['encoding'].values[0];
                w = self.metadata[file]['width_pixels'];
                h = self.metadata[file]['height_pixels'];
                self.labels[file] = self.parse_rle(rle, w, h);
                
        else:
            self.labels = None;
            
        # save transforms
        self.transform = transform;

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist();
        
        file = self.files[idx];
        img = self.read_image(file);
        meta = self.metadata[file];
        
        if self.transform != None:
            img = self.transform(img);
        
        if self.labels != None:
            mask = self.labels[file];
            if self.transform != None:
                mask = self.transform(mask);
                
            return((img, meta, mask));
        else:
            return((img, meta));


    def __len__(self):
        return len(self.files);


    # ======================== helper functions =============================
    def read_csv(self, filename):
        """
        Assumes filename is in projects data/ folder and does pandas.read_csv()
        
        params: filename (string)
        returns: csv_data (dataframe)
        """
        root = os.path.dirname(os.path.abspath(__file__));
        train_file = os.path.join(root, 'data', filename);
        df = pd.read_csv(train_file);
        return(df);


    def parse_rle(self, rle, img_width, img_height):
        """
        Parses run-length encoding and returns sparse matrix with 1s at FTU pixels
        
        params: rle (string) - run length encoding (format: <pixel#> <#of1s> ...)
                img_width (int) - width of image corresponding to rle
                img_height (int) - height of image for rle
        returns: pixel-labelled matrix (scipy.sparse.csr)
        """
        #print("WIDTH:", img_width);
        #print("HEIGHT:", img_height);
        #print("RLE", rle[:30], '...');
        ret = lil_matrix((img_height, img_width), dtype='int8');
        
        rle = rle.strip();
        tokens = rle.split(' ');
        
        for i in range(0, len(tokens), 2):
            pixel = int(tokens[i]);
            num_ones = int(tokens[i+1]);
            for j in range(num_ones):
                y = int(int(pixel+j) / img_width);
                x = int(pixel + j) % img_width;
                #print(f"Setting {int(pixel+j)} => {y},{x}");
                ret[y,x] = 1;
            
        return(ret.tocsr());

    def read_image(self, filename):
        path = os.path.join(self.img_dir, filename);
        img = Image.open(path);
        return( np.array(img) );
    
# ======================= test code if this file is called ====================
if __name__ == "__main__":
    
    trainset = KidneyDataset("data/train", train=True, transform=None);
    
    image, meta, mask = trainset.__getitem__(0);
    
    #fig, ax = plt.subplots(1, 2);
    
    #plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("image")
    plt.imshow(image)
    plt.show();
    
    #plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title("mask")
    plt.imshow(mask.todense())
    plt.show();
    
    print(meta);
    print("mask max:", np.max(mask.todense()));
    print("mask mean:", np.mean(mask.todense()));