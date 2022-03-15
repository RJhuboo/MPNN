import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
from torchvision import transforms, utils
from natsort import natsorted
import argparse



class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0]))
        image = io.imread(img_name) # Loading Image
        #base = np.zeros((RESIZE_IMAGE,RESIZE_IMAGE)) # We need a 512x512 image to be at an order 2n without upscaling^
        #base[6:506,6:506]=image # enelever les chiffres pour des variables
        #image = base # Now, image has 512x512 pixels with a zero border
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        labels = self.labels.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        sample = {'image': image, 'label': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Test_Datasets(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = os.listdir(self.image_dir)
        img_name = os.path.join(self.image_dir,image_name[idx])
        image = io.imread(img_name) # Loading Image
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample
