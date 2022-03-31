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
parser.add_argument("--label_dir", default = "./Label.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "./data/ROI_trab", help = "path to image directory")
parser.add_argument("--train_cross", default = "./cross_output.pkl", help = "filename of the output of the cross validation")
#parser.add_argument("--test_cross",default = "./cross_validation.pkl")
parser.add_argument("--batch_size", type=int, default = 16, help = "number of batch")
parser.add_argument("--model", default = "ConvNet", help="Choose model : Unet or ConvNet") 
parser.add_argument("--nof", type=int, default = 8, help = "number of filter")
parser.add_argument("--lr", type=float, default = 0.001, help = "learning rate")
parser.add_argument("--nb_epochs", type=int, default = 5, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "Train", help = "Mode used : Train, Using or Test")
parser.add_argument("--cross_val", default = False, help = "mode training")
parser.add_argument("--k_fold", type=int, default = 5, help = "Number of splitting for k cross-validation")
parser.add_argument("--n1", type=int, default = 240, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", type=int, default = 120, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", type=int, default = 60, help = "number of neurons in the third layer of the neural network")
parser.add_argument("--nb_workers", type=int, default = 0, help ="number of workers for datasets")
parser.add_argument("--norm_method", type=str, default = "L2", help = "choose how to normalize bio parameters")

opt = parser.parse_args()
PERCENTAGE_TEST = 20
SIZE_IMAGE = 512
NB_LABEL = 14
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
    
### FOR CROSS VALIDATION

def cross_validation():
    # Create the folder where to save results and checkpoints
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/cross"+str(i)) == False:
            save_folder = "./result/cross"+str(i)
            os.mkdir(save_folder)
            break
    
    csv_file = pd.read_csv(opt.label_dir)
    split = train_test_split(csv_file,test_size = 0.2,random_state=1)
    datasets = dataloader.Datasets(csv_file = split[0], image_dir = opt.image_dir, opt=opt) # Create dataset
    if opt.norm_method == "standardization" or opt.norm_method == "minmax":
        scaler = dataloader.normalization(split[0],opt.norm_method)
    else:
        scaler = None
    kf = KFold(n_splits = opt.k_fold, shuffle=True)
    kf.get_n_splits(datasets)
    score_train = []
    score_test = []
    score_mse_t = []
    score_mse_v = []
    print("start cross validation")
    for train_index, test_index in kf.split(datasets):
        print("Train:", train_index[1:4],"Test:",test_index[1:4])
        trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = train_index,  num_workers = opt.nb_workers )
        testloader =DataLoader(datasets, batch_size = 1, sampler = test_index, num_workers = opt.nb_workers )

        # defining the model
        if opt.model == "ConvNet":
            print("## Choose model : convnet ##")
            model = Model.ConvNet(features =opt.nof,out_channels=NB_LABEL,k1 = 3,k2 = 3,k3= 3).to(device)
        elif opt.model == "resnet50":
            print("## Choose model : resnet50 ##")
            model = Model.ResNet50(14,1).to(device)
        elif opt.model == "restnet101":
            print("## Choose model : resnet101 ##")
            model = Model.ResNet101(14,1).to(device)
        else:
            print("## Choose model : Unet ##")
            model = Model.UNet(in_channels=1,out_channels=1,nb_label=NB_LABEL, n1=opt.n1, n2=opt.n2, n3=opt.n3, init_features=opt.nof).to(device)
            model.apply(reset_weights)
        t = Trainer(opt,model,device,save_folder,scaler)
        for epoch in range(opt.nb_epochs):
            [r2_train,mse_train] = t.train(trainloader,epoch)
            [r2_test,mse_test] = t.test(testloader,epoch)
            score_train.append(r2_train)
            score_test.append(r2_test)
            score_mse_t.append(mse_train)
            score_mse_v.append(mse_test)
    resultat = {"r2_train":score_train,"r2_test":score_test,"mse_train":score_mse_t,"mse_test":score_mse_v}
    #torch.save({"loss":score_mse_v}, os.path.join(save_folder,opt.train_cross))
    with open(os.path.join(save_folder,opt.train_cross),'wb') as f:
        pickle.dump(resultat, f)
    with open(os.path.join(save_folder,"history.txt"),'wb') as g:
        history = "nof: " + str(opt.nof) + " model:" +str(opt.model) + " lr:" + str(opt.lr) + " neurons: " + str(opt.n1) + " " + str(opt.n2) + " " + str(opt.n3) + " kernel:" + str(3) + " norm data: " + str(opt.norm_method)
        pickle.dump(history,g)
  
## FOR TRAINING

def train():
    # Create the folder where to save results and checkpoints
    i=1
    if os.path.isdir("./result/train"+str(i)) == False:
        save_folder = "./result/train"+str(i)
        os.mkdir(save_folder)
    while True:
        i += 1
        if os.path.isdir("./result/train"+str(i)) == False:
            save_folder = "./result/train"+str(i)
            os.mkdir(save_folder)
            break

    # defining data
    csv_file = pd.read_csv(opt.label_dir)
    split = train_test_split(data,test_size=0.2,random_state=1)
    testdatasets = dataloader.Datasets(csv_file = split[0], image_dir = opt.image_dir) # Create dataset
    traindatasets = dataloader.Datasets(csv_file = split[1], image_dir = opt.image_dir) # Create dataset
    print("start training")
    trainloader = DataLoader(traindatasets, batch_size = opt.batch_size, num_workers = opt.nb_workers )
    testloader =DataLoader(testdatasets, batch_size = 1, num_workers = opt.nb_workers )

    if opt.norm_method == "standardization" or opt.norm_method == "minmax":
        scaler = dataloader.normalization(split[0],opt.norm_method)
    else:
        scaler = None
    # defining the model
    if opt.model == "ConvNet":
        print("## Choose model : convnet ##")
        model = Model.ConvNet(features =opt.nof,out_channels=NB_LABEL,k1 = 3,k2 = 3,k3= 3).to(device)
    elif opt.model == "resnet50":
        print("## Choose model : resnet50 ##")
        model = Model.ResNet50(14,1).to(device)
    elif opt.model == "restnet101":
        print("## Choose model : resnet101 ##")
        model = Model.ResNet101(14,1).to(device)
    else:
        print("## Choose model : Unet ##")
        model = Model.UNet(in_channels=1,out_channels=1,nb_label=NB_LABEL, n1=opt.n1, n2=opt.n2, n3=opt.n3, init_features=opt.nof).to(device)
        model.apply(reset_weights)
    t = Trainer(opt,model,device,save_folder,scaler)
    for epoch in range(opt.nb_epochs):
        [r2_train,mse_train] = t.train(trainloader,epoch)
        [r2_test,mse_test] = t.test(testloader,epoch)
        score_train.append(r2_train)
        score_test.append(r2_test)
        score_mse_t.append(mse_train)
        score_mse_v.append(mse_test)
    resultat = {"r2_train":score_train,"r2_test":score_test, "mse_train":score_mse_t, "mse_test":score_mse_v}
    with open(os.path.join(save_folder,opt.train_cross),'wb') as f:
        pickle.dump(resultat, f)
    with open(os.path.join(save_folder,"history.txt"),'wb') as g:
        history = "nof: " + str(opt.nof) + " nbbatch:"+ opt.batch_size + " model:" +str(opt.model) + " lr:" + str(opt.lr) + " neurons: " + str(opt.n1) + " " + str(opt.n2) + " " + str(opt.n3) + " kernel:" + str(3) + " norm data: " + str(opt.norm_method) 
        pickle.dump(history,g)
      

''' main '''

if opt.mode == "cross":
  cross_validation()
elif opt.mode == "train":
  train()
  
