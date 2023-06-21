import torch
import os
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
#from skimage.color import rgb2gray
from sklearn import preprocessing
import torchvision.transforms.functional as TF

def normalization(csv_file,indices):
    Data = pd.read_csv(csv_file)
    scaler = preprocessing.StandardScaler()
    scaler.fit(Data.iloc[indices,1:])
    return scaler

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, scaler, opt, upsample = False):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.mask_dir = mask_dir
        self.scaler = scaler
        self.mask_use = True
        self.upsample = upsample
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        mask_name = os.path.join(self.mask_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        image = io.imread(img_name) # Loading Image
        if self.upsample == True or 'lr' in img_name:
            image = transform.rescale(image,2)
            image = (image>0.5)*255
            mask_name = os.path.join(self.mask_dir,str(self.labels.iloc[idx,0]).replace("_lr.tif",".bmp"))
        if self.mask_use == True:
            #mask_name
            mask = io.imread(mask_name)
            mask = transform.rescale(mask, 1/8, anti_aliasing=False)
            print("mask has:", np.unique(mask))
            mask = mask / 255.0 # Normalizing [0;1]
            mask = mask.astype('float32') # Converting images to float32
            #image = rgb2gray(image)
            print("image has:",np.unique(image))
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32
        else:
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32 
        lab = self.scaler.transform(self.labels.iloc[:,1:])
        lab = pd.DataFrame(lab)
        lab.insert(0,"File name", self.labels.iloc[:,0], True)
        lab.columns = self.labels.columns
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
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

        return {'image': image, 'mask': mask, 'label': labels, 'ID': lab.iloc[idx,0]}
