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
    print(len(Data))
    scaler.fit(Data.iloc[indices,1:])
    return scaler

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, opt, indices,transform=None):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
        self.indices = indices
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0])[:-4] + ".png")
        image = io.imread(img_name) # Loading Image
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        if self.opt.norm_method== "L2":
            lab = preprocessing.normalize(self.labels.iloc[:,1:],axis=0)
        elif self.opt.norm_method == "L1":
            lab = preprocessing.normalize(self.labels.iloc[:,1:],norm='l1',axis=0)
        elif self.opt.norm_method == "minmax":
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(self.labels.iloc[self.indices,1:])
            lab = scaler.transform(self.labels.iloc[:,1:])
        elif self.opt.norm_method == "standardization":
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.labels.iloc[self.indices,1:])
            lab = scaler.transform(self.labels.iloc[:,1:])
        lab = pd.DataFrame(lab)
        lab.insert(0,"File name", self.labels.iloc[:,0], True)
        lab.columns = self.labels.columns
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        if self.transform:
            sample = self.transform(sample)
        return {'image': image, 'label': labels, 'ID': lab.iloc[idx,0]}

class Test_Datasets(Dataset):
    def __init__(self, csv_file, image_dir, opt, scaler, transform=None):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
        self.scaler = scaler
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0])[:-4] + ".png")
        print(img_name)
        image = io.imread(img_name) # Loading Image
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        lab = self.scaler.transform(self.labels.iloc[:,1:])
        lab = pd.DataFrame(lab)
        lab.insert(0,"File name", self.labels.iloc[:,0], True)
        lab.columns = self.labels.columns
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        if self.transform:
            sample = self.transform(sample)
        return {'image': image, 'label': labels, 'ID': lab.iloc[idx,0]}
   
