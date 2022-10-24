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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import Model
from trainer import Trainer
import dataloader

# GPU or CPU
if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "./Label_6p.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/MOUSE_BPNN/HR/Train_Label_trab", help = "path to image directory")
parser.add_argument("--train_cross", default = "./cross_multinet.pkl", help = "filename of the output of the cross validation")
parser.add_argument("--batch_size", type=int, default = 16, help = "number of batch")
parser.add_argument("--model", default = "MultiNet", help="Choose model : Unet or ConvNet") 
parser.add_argument("--nof", type=int, default = 8, help = "number of filter")
parser.add_argument("--lr", type=float, default = 0.005, help = "learning rate")
parser.add_argument("--nb_epochs", type=int, default = 120, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "Train", help = "Mode used : Train, Using or Test")
parser.add_argument("--k_fold", type=int, default = 5, help = "Number of splitting for k cross-validation")
parser.add_argument("--n1", type=int, default = 240, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", type=int, default = 120, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", type=int, default = 60, help = "number of neurons in the third layer of the neural network")
parser.add_argument("--nb_workers", type=int, default = 0, help ="number of workers for datasets")
parser.add_argument("--norm_method", type=str, default = "standardization", help = "choose how to normalize bio parameters")
parser.add_argument("--NB_LABEL", type=int, default = 6, help = "specify the number of labels")
parser.add_argument("--nb_hidden_layer", default = 50, help= "Number of hidden layer")
parser.add_argument("--task_specific_hidden_size", default= 100, help="number of neurons in specific layers")
parser.add_argument("--hidden_size", default= 100, help="number of neurons in hard sharing layers")

opt = parser.parse_args()
NB_DATA = 4073
PERCENTAGE_TEST = 20
SIZE_IMAGE = 512
NB_LABEL = opt.NB_LABEL
'''functions'''

## RESET WEIGHT FOR CROSS VALIDATION

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

## FOR TRAINING

def train():
    # Create the folder where to save results and checkpoints
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/train"+str(i)) == False:
            save_folder = "./result/train"+str(i)
            os.mkdir(save_folder)
            break
    score_mse_t = []
    score_mse_v = []
    # defining data
    index = range(NB_DATA)
    split = train_test_split(index,test_size = 0.2,shuffle=False)
    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, opt=opt, indices = split[0]) # Create dataset
    print("start training")
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = shuffle(split[0]), num_workers = opt.nb_workers )
    testloader =DataLoader(datasets, batch_size = 1, sampler = shuffle(split[1]), num_workers = opt.nb_workers )

    if opt.norm_method == "standardization" or opt.norm_method == "minmax":
        scaler = dataloader.normalization(opt.label_dir,opt.norm_method,split[0])
    else:
        scaler = None
    # defining the model
    if opt.model == "ConvNet":
        print("## Choose model : convnet ##")
        model = Model.ConvNet(features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    elif opt.model == "MultiNet":
        print("## Choose model : MultiNet ##")
        model = Model.HardSharing(input_size=64*64*64,hidden_size=opt.hidden_size,n_hidden=opt.nb_hidden_layer,n_outputs=NB_LABEL,task_specific_hidden_size=opt.task_specific_hidden_size).to(device)

    #model.apply(reset_weights)
    
    # Start training
    t = Trainer(opt,model,device,save_folder,scaler)
    for epoch in range(opt.nb_epochs):
        mse_train = t.train(trainloader,epoch)
        mse_test = t.test(testloader,epoch)
        score_mse_t.append(mse_train)
        score_mse_v.append(mse_test)
    resultat = {"mse_train":score_mse_t, "mse_test":score_mse_v}
    with open(os.path.join(save_folder,opt.train_cross),'wb') as f:
        pickle.dump(resultat, f)
    with open(os.path.join(save_folder,"history.txt"),'wb') as g:
        history = "nof: " + str(opt.nof) + " model:" +str(opt.model) + " lr:" + str(opt.lr) + " neurons: " + str(opt.n1) + " " + str(opt.n2) + " " + str(opt.n3) + " kernel:" + str(3) + " norm data: " + str(opt.norm_method)
        pickle.dump(history,g)
      

''' main '''

train()
  
