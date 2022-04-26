import torch
import os
import numpy as np
import pandas as pd
import argparse
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pickle
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

def MSE(y_predicted,y):
    squared_error = (y_predicted - y) **2
    sum_squared_error = np.sum(squared_error)
    mse = sum_squared_error / y.size
    return mse

class Trainer():
    def __init__(self,opt,my_model,device,save_fold,scaler):
        self.scaler = scaler
        self.save_fold = save_fold
        self.device = device
        self.opt = opt
        self.model = my_model
        self.NB_LABEL = opt.NB_LABEL
        self.optimizer = Adam(self.model.parameters(), lr=self.opt.lr)
        self.criterion = MSELoss()
        
    def train(self, trainloader, epoch ,steps_per_epochs=20):
        self.model.train()
        print("starting training")
        print("----------------")
        train_loss = 0.0
        train_total = 0
        running_loss = 0.0
        mse_score = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data['image'], data['label']
            
            # reshape
            inputs = inputs.reshape(inputs.size(0),1,512,512)
            labels = labels.reshape(labels.size(0),self.NB_LABEL)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            # forward backward and optimization
            outputs = self.model(inputs)
            if self.opt.model == "MultiNet:
                loss1 = self.criterion(outputs[0],torch.reshape(labels[0],[len(outputs[0]),1]))
                loss2 = self.criterion(outputs[1],torch.reshape(labels[1],[len(outputs[1]),1]))
                loss3 = self.criterion(outputs[2],torch.reshape(labels[2],[len(outputs[2]),1]))
                loss4 = self.criterion(outputs[3],torch.reshape(labels[3],[len(outputs[3]),1]))
                loss5 = self.criterion(outputs[4],torch.reshape(labels[4],[len(outputs[4]),1]))
                loss = (self.opt.alpha1*loss1) + (self.opt.alpha2*loss2) + (self.opt.alpha3*loss3) + (self.opt.alpha4*loss4) + (self.opt.alpha5*loss5)
            else:
                loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            
            # statistics
            train_loss += loss.item()
            running_loss += loss.item()
            train_total += 1
            #outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            #labels, outputs = np.array(labels), np.array(outputs)
            #labels, outputs = labels.reshape(self.NB_LABEL,len(inputs)), outputs.reshape(self.NB_LABEL,len(inputs))
            if i % self.opt.batch_size == self.opt.batch_size-1:
                print('[%d %5d], loss: %.3f' %
                      (epoch + 1, i+1, running_loss/self.opt.batch_size))
                running_loss = 0.0
                
        # displaying results
        mse = train_loss / train_total
        print('Epoch [{}], Loss: {}'.format(epoch+1, train_loss/train_total), end='')
        print('Finished Training')
        
        # saving trained model
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        torch.save(self.model.state_dict(),os.path.join(self.opt.checkpoint_path,check_name))
        return mse

    def test(self,testloader,epoch):
        self.model.eval()

        test_loss = 0
        test_total = 0
        mse_score = 0.0
        output = {}
        label = {}
        
        # Loading Checkpoint
        if self.opt.mode == "Test":
            check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
            self.model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        
        # Testing
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data['image'],data['label']
                
                # reshape
                inputs = inputs.reshape(1,1,512,512)
                labels = labels.reshape(1,self.NB_LABEL)
                inputs, labels = inputs.to(self.device),labels.to(self.device)
                
                # loss
                outputs = self.model(inputs)
                if self.opt.model == "MultiNet:
                    loss1 = self.criterion(outputs[0],torch.reshape(labels[0],[1,1]))
                    loss2 = self.criterion(outputs[1],torch.reshape(labels[1],[1,1]))
                    loss3 = self.criterion(outputs[2],torch.reshape(labels[2],[1,1]))
                    loss4 = self.criterion(outputs[3],torch.reshape(labels[3],[1,1]))
                    loss5 = self.criterion(outputs[4],torch.reshape(labels[4],[1,1]))
                    loss = (self.opt.alpha1*loss1) + (self.opt.alpha2*loss2) + (self.opt.alpha3*loss3) + (self.opt.alpha4*loss4) + (self.opt.alpha5*loss5)
                else:
                    loss = self.criterion(outputs,labels)
                test_loss += loss.item()
                test_total += 1
                
                # statistics
                #outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
                #labels, outputs = np.array(labels), np.array(outputs)
                #labels, outputs = labels.reshape(self.NB_LABEL,1), outputs.reshape(self.NB_LABEL,1)
                #outputs,labels=outputs.reshape(1,self.NB_LABEL), labels.reshape(1,self.NB_LABEL)
                if self.opt.norm_method == "standardization" or self.opt.norm_method == "minmax":
                    outputs,labels = self.scaler.inverse_transform(outputs), self.scaler.inverse_transform(labels)
                output[i] = outputs
                label[i] = labels
            name_out = "./output" + str(epoch) + ".txt"
            name_lab = "./label" + str(epoch) + ".txt"
            mse = test_loss/test_total
            with open(os.path.join(self.save_fold,name_out),"wb") as f:
                pickle.dump(output,f)
            with open(os.path.join(self.save_fold,name_lab),"wb") as f:
                pickle.dump(label,f)
           
        print(' Test_loss: {}'.format(test_loss/test_total))
        return mse
