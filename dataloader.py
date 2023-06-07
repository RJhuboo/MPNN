import torch
import os
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from skimage import io,transform
from sklearn import preprocessing
import torchvision.transforms.functional as TF
from skimage import morphology


# Normalization function for morphometry data
def normalization(csv_file,mode,indices):
    Data = pd.read_csv(csv_file)
    # Initialize scaler
    if mode == "standardization":
        scaler = preprocessing.StandardScaler()
    elif mode == "minmax":
        scaler = preprocessing.MinMaxScaler()
    # Compute mean and standard deviation
    scaler.fit(Data.iloc[indices,1:])
    return scaler

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, scaler, opt):
        """ Initializes the datasets variables.

        Args:
            csv_file (_type_): Label csv file 
            image_dir (_type_): directory of the images
            mask_dir (_type_): directory of the masks
            scaler (_type_): normalization informations
            opt (_type_): Some options
        """
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.scaler=scaler
        self.mask_dir = mask_dir
        self.mask_use = True # Tune for use of mask
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Find image path
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        mask_name = os.path.join(self.mask_dir, str(self.labels.iloc[idx,0][:-4] + ".bmp"))
        
        # Read image and mask
        image = io.imread(img_name)
        if 'lr' in img_name: # If image is a low resolution image
            image = transform.rescale(image,2) # Rescaling the image to match size of high resolution image
            image = (image<0.5)*255 # Binarized the Image between 0 and 255
            mask_name = os.path.join(self.mask_dir,str(self.labels.iloc[idx,0]).replace("_lr.tif",".bmp")) # Find the corresponding mask file
        if self.mask_use == True:
            mask = io.imread(mask_name) # Read the mask
            mask = transform.rescale(mask, 1/8, anti_aliasing=False) # Rescaling the mask
            mask = mask / 255.0 # Normalizing [0;1]
            mask = mask.astype('float32') # Converting images to float32
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32
        else:
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32 
        
        skel,dist = morphology.medial_axis(image) # Find the medial axis of the image
        skel.astype('float32') # Converting images to float32
        dist.astype('float32') # Converting images to float32
        lab = self.scaler.transform(self.labels.iloc[:,1:]) # Apply the normalization to labels
        lab = pd.DataFrame(lab) # Converting labels to pandas dataframe
        lab.insert(0,"File name", self.labels.iloc[:,0], True) # Inset the name of images
        lab.columns = self.labels.columns # Take the columns names
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) # Converting labels to numpy array
        labels = labels.astype('float32') # Converting labels to float32
        
        # Image transformation for data augmentation
        p = random.random()
        rot = random.randint(-45,45)
        transform_list = []
        image,mask,skel,dist=TF.to_pil_image(image),TF.to_pil_image(mask),TF.to_pil_image(skel),TF.to_pil_image(dist)
        image,mask,skel,dist=TF.rotate(image,rot),TF.rotate(mask,rot),TF.rotate(skel,rot),TF.rotate(dist,rot)
        if p<0.3:
            image,mask,skel,dist=TF.vflip(image),TF.vflip(mask),TF.vflip(skel),TF.vflip(dist)
        p = random.random()
        if p<0.3:
            image,mask,skel,dist=TF.hflip(image),TF.hflip(mask),TF.hflip(skel),TF.hflip(dist)
        p = random.random()
        if p>0.2:
            image,mask,skel,dist=TF.affine(image,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(mask,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(skel,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(dist,angle=0,translate=(0.1,0.1),shear=0,scale=1)
        image,mask,skel,dist=TF.to_tensor(image),TF.to_tensor(mask),TF.to_tensor(skel),TF.to_tensor(dist)
        
        return {'image': image,'mask':mask, 'label': labels, 'skel':skel, 'dist':dist}