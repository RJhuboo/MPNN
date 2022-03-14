import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io,transform
from torchvision import transforms, utils
from torchvision.utils import make_grid
from natsort import natsorted
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import random
import pickle

''' Training function '''

def train(model, train_loader, optimizer, criterion, epoch, opt, steps_per_epochs=20):
    model.train()
    print("starting training")
    print("----------------")
    train_loss = 0.0
    train_total = 0
    running_loss = 0.0
    r2_s = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data['image'], data['label']
        # reshape
        inputs = inputs.reshape(inputs.size(0),1,512,512)
        labels = labels.reshape(labels.size(0),NB_LABEL)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward backward and optimization
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        # statistics
        train_loss += loss.item()
        running_loss += loss.item()
        train_total += labels.size(0)
        outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        labels, outputs = np.array(labels), np.array(outputs)
        labels, outputs = labels.reshape(34,opt.batch_size), outputs.reshape(34,opt.batch_size)
        r2 = r2_score(labels,outputs)
        r2_s += r2
        
        if i % opt.batch_size == opt.batch_size-1:
            print('[%d %5d], loss: %.3f' %
                  (epoch + 1, i+1, running_loss/opt.batch_size))
            running_loss = 0.0
    # displaying results
    r2_s = r2_s/i
    print('Epoch [{}], Loss: {}, R square: {}'.format(epoch+1, train_loss/train_total, r2_s), end='')
    
    print('Finished Training')
    # saving trained model
    check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
    torch.save(model.state_dict(),os.path.join(opt.checkpoint_path,check_name))

''' Testing function '''

def test(model, test_loader, criterion, epoch, opt):
    model.eval()
    
    test_loss = 0
    test_total = 0
    r2_s = 0
    output = {}
    label = {}
    # Loading Checkpoint
    if opt.mode is "Test":
        model = Net()
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path,check_name)))
    # Testing
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data['image'],data['label']
            # reshape
            inputs = inputs.reshape(1,1,512,512)
            labels = labels.reshape(1,NB_LABEL)
            # loss
            outputs = model(inputs)
            test_loss += criterion(outputs,labels)
            test_total += labels.size(0)
            # statistics
            outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            labels, outputs = np.array(labels), np.array(outputs)
            labels, outputs = labels.reshape(34,1), outputs.reshape(34,1)
            r2 = r2_score(labels,outputs)
            r2_s += r2
            print('r2 : %.3f , MSE : %.3f' %
                  (r2,test_loss))
            output[i] = outputs
            label[i] = labels
        name_out = "./output" + str(epoch) + ".txt"
        name_lab = "./label" + str(epoch) + ".txt"

        with open(name_out,"wb") as f:
            pickle.dump(output,f)
        with open(name_lab,"wb") as f:
            pickle.dump(label,f)
    
    r2_s = r2_s/i
    print(' Test_loss: {}, Test_R_square: {}'.format(test_loss/test_total, r2_s))
    if opt.mode == "Train":
        # wandb.log({'Test Loss': test_loss/test_total, 'Test R square': r2_s})

""" Function of usage """

def using_model(model, test_loader,opt):
    model.eval()
    epoch = 4
    output = {}
    label = {}
    # Loading Checkpoint
    model = Net()
    check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
    model.load_state_dict(torch.load(os.path.join(opt.checkpoint_path,check_name)))
    # Testing
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data['image']
            # reshape
            inputs = inputs.reshape(1,1,512,512)
            # loss
            outputs = model(inputs)
            # statistics
            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(34,1)
            
        name_out = "./output" + str(epoch) + ".txt"

        with open(name_out,"wb") as f:
            pickle.dump(output,f)
