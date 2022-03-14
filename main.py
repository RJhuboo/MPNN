import torch
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import KFold

import random
#import wandb

import Model
from trainer import Trainer
import dataloader

# GPU or CPU
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/media/rehan/Seagate Expansion Drive/sainbiose/New_datasets/HR_TIF_NOAUGMENTATION/Labels_norm_aug.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/media/rehan/Seagate Expansion Drive/sainbiose/New_datasets/HR_PNG_AUGMENTATION", help = "path to image directory")
parser.add_argument("--batch_size", default = 16, help = "number of batch")
parser.add_argument("--nof", default = 16, help = "number of filter")
parser.add_argument("--lr",default = 0.001, help = "learning rate")
parser.add_argument("--nb_epochs", default = 5, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "/media/rehan/Seagate Expansion Drive/src/nodropout", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "Using", help = "Mode used : Training, Using or Testing")
parser.add_argument("--cross_val", default = False, help = "mode training")
parser.add_argument("--k_fold", default = 5, help = "number of splitting for k cross-validation")


opt = parser.parse_args()
NB_LABEL = 34
PERCENTAGE_TEST = 20
RESIZE_IMAGE = 512

''' main '''

# defining data
if opt.mode == "Train" or opt.mode == "Test":
    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir) # Create dataset
else:
    datasets = dataloader.Datasets(image_dir = opt.image_dir)
# defining the model
model = Model.Net()

# DATA SPLITING - Two possibilities : split data randomly or split data with specific index 
if opt.mode == "Train" or opt.mode == "Test":
    nb_data = len(datasets)
    nb_test = nb_data*(PERCENTAGE_TEST/100)
    nb_dataset = nb_data - nb_test
    # train,test = torch.utils.data.random_split(datasets, [int(nb_dataset),int(nb_test)]) # split data randomly
    ids = np.array(range(0,nb_data))
    random.shuffle(ids)
    train_ids = ids[0:int(nb_dataset)]
    test_ids = ids [int(nb_dataset):]
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = train_ids,  num_workers = 0 ) # Create batches and tensors
    testloader = DataLoader(datasets, batch_size = 1, sampler = test_ids, num_workers = 0 ) # Create batches and tensors
    # trainloader = DataLoader(train, batch_size = opt.batch_size, shuffle = True, num_workers = 0 ) # Create batches and tensors
    # testloader = DataLoader(test, batch_size = 1, shuffle = False, num_workers = 0 ) # Create batches and tensors
else:
    testloader = DataLoader(datasets,batch_size = 1, num_workers =0)
# training 

if opt.mode == "Train" or opt.mode == "Test":
    #wandb.init(entity='jhuboo', project='BPNN1')
    #wandb.watch(model, log='all')
    t = trainer.Trainer(opt,model)
    for epoch in range(opt.nb_epochs):
        t.train(trainloader,epoch)
        t.test(testloader,epoch)    
    #wandb.finish()
else:
    t = trainer.Trainer(opt,model)
    t.test(testloader,4)
