import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
from torchvision import transforms, utils
import argparse
from sklearn import preprocessing

def normalization(csv_file,mode,indices):
    Data = pd.read_csv(csv_file)
    if mode == "standardization":
        scaler = preprocessing.StandardScaler()
    elif mode == "minmax":
        scaler = preprocessing.MinMaxScaler()
    scaler.fit(Data.iloc[indices,1:])
    return scaler

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, scaler, opt,transform=None):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.mask_dir = mask_dir
        self.transform = transform
        self.scaler = scaler
        self.mask_use = False
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        mask_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".bmp"))
        image = io.imread(img_name) # Loading Image
        if self.mask_use == True:
            mask = io.imread(mask_name)
            mask = mask / 255.0 # Normalizing [0;1]
            mask = mask.astype('float32') # Converting images to float32
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32
            image = np.dstack( ( image, mask ) )
        else:
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32 
        lab = self.scaler.transform(self.labels.iloc[:,1:])
        lab = pd.DataFrame(lab)
        lab.insert(0,"File name", self.labels.iloc[:,0], True)
        lab.columns = self.labels.columns
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        #print(np.shape(labels))
        #labels = labels.reshape(-1,1)
        labels = labels.astype('float32')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': labels, 'ID': lab.iloc[idx,0]}

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
        #sample = {'image': image}
        #if self.transform:
        #    sample = self.transform(sample)
        return image
