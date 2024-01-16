import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
from torchvision import transforms, utils
import argparse
from sklearn import preprocessing
import torchvision.transforms.functional as TF
import random

# Normalization function
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
    def __init__(self, csv_file, image_dir, mask_dir, scaler, opt):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.mask_dir = mask_dir
        self.scaler = scaler
        self.mask_use = True
        self.upsample = False
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Image loading
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        image = io.imread(img_name) 
        image = image.astype('float32') # Converting images to float32

        # If data is Low Resolution then upsample the image
        if self.upsample == True or 'lr' in img_name:
            image = (image>0.5)*1
            mask_name = os.path.join(self.mask_dir,(str(self.labels.iloc[idx,0]).replace(".tif",".png")).replace("im_lr_","im"))

        # Mask loading
        if self.mask_use == True:
            mask_name = os.path.join(self.mask_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
            mask = io.imread(mask_name)
            mask = (transform.rescale(mask, 1/8, anti_aliasing=False)>0.1)*1.
            mask = mask.astype('float32') # Converting images to float32

        # Normalize the biological parameters
        lab = self.scaler.transform(self.labels.iloc[:,1:])
        labels = np.array([lab]) 
        labels = labels.astype('float32')

        # Data augmentation (Applied on both mask and image)
        p = random.random()
        rot = random.randint(-45,45)
        transform_list = []
        image,mask=TF.to_pil_image(image),TF.to_pil_image(mask)
        image,mask=TF.rotate(image,rot),TF.rotate(mask,rot)
        if p<0.3:
            image,mask=TF.vflip(image),TF.vflip(mask)
        p = random.random()
        if p<0.3:
            image,mask=TF.hflip(image),TF.hflip(mask)
        p = random.random()
        if p>0.2:
            image,mask=TF.affine(image,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(mask,angle=0,translate=(0.1,0.1),shear=0,scale=1)
        image,mask=TF.to_tensor(image),TF.to_tensor(mask)
        
        return {'image': image, 'mask': mask, 'label': labels, 'ID':  self.labels.iloc[idx,0]}
   
