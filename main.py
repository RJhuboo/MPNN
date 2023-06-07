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
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import Model
from trainer import Trainer
import dataloader
import optuna
import joblib

# GPU or CPU
if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/gpfswork/rech/tvs/uki75tv/BPNN/csv_files/Train_Label_7p_lrhr.csv", help = "path to label csv file")  #"./Train_Label_7p_lrhr.csv")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_LR_segmented", help = "path to image directory")  #"./Train_LR_segmented")"
parser.add_argument("--mask_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_trab_mask", help = "path to mask")
parser.add_argument("--in_channel", type=int, default = 1, help = "nb of image channel")
parser.add_argument("--train_cross", default = "./cross_output.pkl", help = "filename of the output of the cross validation")
parser.add_argument("--batch_size", type=int, default = 1, help = "number of batch")
parser.add_argument("--model", default = "ConvNet", help="Choose model : Unet or ConvNet") 
parser.add_argument("--nof", type=int, default = 64, help = "number of filter")
parser.add_argument("--lr", type=float, default = 0.000123, help = "learning rate")
parser.add_argument("--nb_epochs", type=int, default = 1000, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "train", help = "Mode used : Train, Using or Test")
parser.add_argument("--k_fold", type=int, default = 1, help = "Number of splitting for k cross-validation")
parser.add_argument("--n1", type=int, default = 158, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", type=int, default = 152, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", type=int, default = 83, help = "number of neurons in the third layer of the neural network")
parser.add_argument("--nb_workers", type=int, default = 0, help ="number of workers for datasets")
parser.add_argument("--norm_method", type=str, default = "standardization", help = "choose how to normalize bio parameters")
parser.add_argument("--NB_LABEL", type=int, default = 7, help = "specify the number of labels")
parser.add_argument("--optim", type=str, default = "Adam", help= "specify the optimizer")

opt = parser.parse_args()
NB_DATA = 600
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
    save_folder=None
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/mouse_multi"+str(i)) == False:
            save_folder = "./result/mouse_multi"+str(i)
            os.mkdir(save_folder)
            break
    score_mse_t = []
    score_mse_v = []
    score_train_per_param = []
    score_test_per_param = []
    # defining data
    index = range(NB_DATA)
    #index_set = train_test_split(index,test_size=100,shuffle=False)
    index_set=train_test_split(index,test_size=0.4,random_state=42)
    scaler = dataloader.normalization(opt.label_dir,opt.norm_method,index_set[0])
    #scaler = dataloader.normalization("/gpfswork/rech/tvs/uki75tv/BPNN/csv_files/Train_Label_7p_lrhr.csv",opt.norm_method,range(10500))

    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, mask_dir = opt.mask_dir, scaler=scaler, opt=opt) # Create dataset
    print("start training")
    
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = shuffle(index_set[0]), num_workers = opt.nb_workers )
    testloader = DataLoader(datasets, batch_size = 1, num_workers = opt.nb_workers,sampler=index_set[1])#, shuffle=True)

    model = Model.ConvNet(in_channel=opt.in_channel,features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)

    torch.manual_seed(2)
    #model.apply(reset_weights)
    t = Trainer(opt,model,device,save_folder,scaler=None)
    for epoch in range(opt.nb_epochs):
        mse_train, param_train = t.train(trainloader,epoch)
        mse_test, param_test = t.test(testloader,epoch)
        score_mse_t.append(mse_train)
        score_mse_v.append(mse_test)
        score_train_per_param.append(param_train)
        score_test_per_param.append(param_test)
    resultat = {"mse_train":score_mse_t, "mse_test":score_mse_v,"train_per_param":score_train_per_param,"test_per_param":score_test_per_param}
    with open(os.path.join(save_folder,opt.train_cross),'wb') as f:
        pickle.dump(resultat, f)
    with open(os.path.join(save_folder,"history.txt"),'wb') as g:
        history = "nof: " + str(opt.nof) + " model:" +str(opt.model) + " lr:" + str(opt.lr) + " neurons: " + str(opt.n1) + " " + str(opt.n2) + " " + str(opt.n3) + " kernel:" + str(3) + " norm data: " + str(opt.norm_method)
        pickle.dump(history,g)
      
''' main '''
if opt.mode == "train":
    train()
else :
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/test_FSRCNN_"+str(i)) == False:
            save_folder = "./result/test_FSRCNN_"+str(i)
            os.mkdir(save_folder)
            break
    
    # model #
    index = list(range(NB_DATA))
    scaler = dataloader.normalization(opt.label_dir,opt.norm_method,index)
    datasets = dataloader.Datasets(csv_file = "./Label_trab_FSRCNN.csv", image_dir="./TRAB_FSRCNN", mask_dir = "./MASK_FSRCNN", scaler=scaler,opt=opt, upsample=False)
    #datasets = dataloader.Datasets(csv_file = "./Test_Label_6p.csv", image_dir="/gpfsstore/rech/tvs/uki75tv/Test_segmented_filtered", mask_dir = "/gpfsstore/rech/tvs/uki75tv/Test_trab_mask", scaler=scaler,opt=opt, upsample=False)
    #index_human = range(400)
    #index_set=train_test_split(index_human,test_size=0.90,random_state=42)
    model = Model.ConvNet(in_channel=opt.in_channel,features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    #scaler = dataloader.normalization("./Train_Label_6p_augment.csv", opt.norm_method,index)
    testloader = DataLoader(datasets, batch_size = 1, num_workers = opt.nb_workers)
    t = Trainer(opt,model,device,save_folder,scaler)
    t.test(testloader,opt.nb_epochs)
  
