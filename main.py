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
#import wandb

import Model
from trainer import Trainer
import dataloader

# GPU or CPU
if torch.cuda.is_available():  
  device = "cuda:0" 
else:  
  device = "cpu"
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "./Label.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "./data/HR_trab", help = "path to image directory")
parser.add_argument("--batch_size", default = 16, help = "number of batch")
parser.add_argument("--nof", default = 16, help = "number of filter")
parser.add_argument("--lr",default = 0.001, help = "learning rate")
parser.add_argument("--nb_epochs", default = 5, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "Train", help = "Mode used : Train, Using or Test")
parser.add_argument("--cross_val", default = False, help = "mode training")
parser.add_argument("--k_fold", default = 5, help = "number of splitting for k cross-validation")
parser.add_argument("--n1", default = 240, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", default = 120, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", default = 60, help = "number of neurons in the third layer of the neural network")
                    
PERCENTAGE_TEST = 20
SIZE_IMAGE = 512
NB_LABEL = 14

''' main '''

# defining data
if opt.mode == "Train" or opt.mode == "Test":
    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir) # Create dataset
else:
    datasets = dataloader.Datasets(image_dir = opt.image_dir)
# defining the model
model = Model.Net(opt.nof,NB_LABEL)

if opt.mode == "Train" or opt.mode == "Test":
    kf = KFold(n_splits = opt.k_fold, shuffle=True)
    kf.get_n_splits(datasets)
    score_train = []
    score_test = []
    for train_index, test_index in kf.split(datasets):
        print("Train:", train_index[1:4],"Test:",test_index[1:4])
        nb_data = len(datasets)
        trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = train_index,  num_workers = 0 )
        testloader =DataLoader(datasets, batch_size = 1, sampler = test_index, num_workers = 0 )
        t = Trainer(opt,model)
        for epoch in range(opt.nb_epochs):
            score_train.append(t.train(trainloader,epoch))
            score_test.append(t.test(testloader,epoch))
    with open('cross_val.pickle','wb') as f:
        pickle.dump(score_train, f)
        pickle.dump(score_test,f)

else:
    testloader = DataLoader(datasets,batch_size = 1, num_workers =0)
