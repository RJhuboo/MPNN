import torch
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


import torch.optim as optim
import torchvision
from torchvision import datasets,transforms 

from skorch import NeuralNetClassifier

from sklearn.model_selection import train_test_split
import Model
from trainer import Trainer
import dataloader

class Args:
    label_dir = "./Label_5p.csv"
    image_dir = "./data/ROI_trab"
    train_cross = "./cross_output.pkl"
    batch_size = 8
    model = "ConvNet" 
    nof = 8
    lr= 0.001
    nb_epochs = 5
    checkpoint_path= "./"
    mode= "Train"
    cross_val = False
    k_fold= 5
    n1= 240
    n2= 120
    n3 = 60
    nb_workers = 0
    norm_method="standardization"

opt = Args()
NB_DATA = 3991
NB_LABEL = 5
PERCENTAGE_TEST = 20
RESIZE_IMAGE = 512

if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
  
net = NeuralNetClassifier(
    Model.ConvNet(features = opt.nof,out_channels=NB_LABEL),
    max_epochs = 10,
    iterator_train__num_workers=4,
    iterator_valid__num_workers=4,
    lr=0.001,
    batch_size=8,
    optimizer=optim.Adam,
    criterion=torch.nn.MSELoss,
    device=device
)

index = range(NB_DATA)
split = train_test_split(index,test_size = 0.2,random_state=1)
scaler = dataloader.normalization(opt.label_dir,opt.norm_method,split[0])
datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, opt=opt, indices =split[0]) # Create dataset
trainloader = DataLoader(datasets, num_workers = 4)
#y_train = np.array([data['label'] for i,data in enumerate(datasets)])
#y_train = np.resize(y_train,[3991,5])
#x_train = np.array([data['image'] for i,data in enumerate(datasets)])
net.fit(datasets)
