import torch
import os
import numpy as np
import pandas as pd
from natsort import natsorted
import argparse
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pickle

class Trainer():
    def __init__(self,opt,my_model):
        self.opt = opt
        self.model = my_model
        self.NB_LABEL = 14
        self.optimizer = Adam(self.model.parameters(), lr=self.opt.lr)
        self.criterion = MSELoss()
        
    def train(self, trainloader, epoch ,steps_per_epochs=20):
        self.model.train()
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
            labels = labels.reshape(labels.size(0),self.NB_LABEL)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # forward backward and optimization
            outputs = self.model(inputs)
            loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            # statistics
            train_loss += loss.item()
            running_loss += loss.item()
            train_total += labels.size(0)
            outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            labels, outputs = np.array(labels), np.array(outputs)
            labels, outputs = labels.reshape(self.NB_LABEL,self.opt.batch_size), outputs.reshape(self.NB_LABEL,self.opt.batch_size)
            r2 = r2_score(labels,outputs)
            r2_s += r2

            if i % self.opt.batch_size == self.opt.batch_size-1:
                print('[%d %5d], loss: %.3f' %
                      (epoch + 1, i+1, running_loss/self.opt.batch_size))
                running_loss = 0.0
        # displaying results
        r2_s = r2_s/i
        print('Epoch [{}], Loss: {}, R square: {}'.format(epoch+1, train_loss/train_total, r2_s), end='')

        print('Finished Training')
        # saving trained model
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        torch.save(self.model.state_dict(),os.path.join(self.opt.checkpoint_path,check_name))

    def test(self,testloader,epoch):
        self.model.eval()

        test_loss = 0
        test_total = 0
        r2_s = 0
        output = {}
        label = {}
        # Loading Checkpoint
        if self.opt.mode is "Test":
            model = self.model
            check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
            model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        # Testing
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data['image'],data['label']
                # reshape
                inputs = inputs.reshape(1,1,512,512)
                labels = labels.reshape(1,self.NB_LABEL)
                # loss
                outputs = model(inputs)
                test_loss += criterion(outputs,labels)
                test_total += labels.size(0)
                # statistics
                outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
                labels, outputs = np.array(labels), np.array(outputs)
                labels, outputs = labels.reshape(self.NB_LABEL,1), outputs.reshape(self.NB_LABEL,1)
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
        if self.opt.mode == "Train":
            print("noting")
            # wandb.log({'Test Loss': test_loss/test_total, 'Test R square': r2_s})
