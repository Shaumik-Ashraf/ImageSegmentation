# -*- coding: utf-8 -*-
# Pytorch dataset implementation for HuBMAP 


import torch;
import os;
import glob;
import pandas as pd;
from torch.utils.data import Dataset;
from scipy import sparse;

class KidneyDataset(Dataset):
    
    def __init__(csv_file, img_dir, train = True, transform = None):
        """
        Parameters
        ----------
        csv_file : TYPE
            DESCRIPTION.
        img_dir : TYPE
            DESCRIPTION.
        train : TYPE, optional
            DESCRIPTION. The default is True.
        transform : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        """
        self.files = [];
        for file in os.listdir(img_dir):
            if( file.find('.tiff') > 0 ):
                self.files.append(file);
        
        if train:
            df = pd.read_csv(csv_file);
            
        else:

