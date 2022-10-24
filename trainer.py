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
        self.optimizer = SGD(self.model.parameters(), lr=self.opt.lr)
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
            if self.opt.model == "MultiNet":
                loss = self.criterion(outputs,labels)
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
                print("--- OUTPUTS ---")
                print(outputs[:8])
                print("--- LABELS ---")
                print(labels[:8])
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
        IDs = {}
        # Loading Checkpoint
        if self.opt.mode == "Test":
            check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
            self.model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        
        # Testing
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels, ID = data['image'],data['label'],data['ID']
                # reshape
                inputs = inputs.reshape(1,1,512,512)
                labels = labels.reshape(1,self.NB_LABEL)
                inputs, labels = inputs.to(self.device),labels.to(self.device)
                
                # loss
                outputs = self.model(inputs)
                if self.opt.model == "MultiNet":
                    loss = self.criterion(outputs,labels)
                else:
                    loss = self.criterion(outputs,labels)
                test_loss += loss.item()
                test_total += 1
                
                # statistics
                labels, outputs = labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()
                labels, outputs = np.array(labels), np.array(outputs)
               
                #labels, outputs = labels.reshape(self.NB_LABEL,1), outputs.reshape(self.NB_LABEL,1)
                labels=labels.reshape(1,self.NB_LABEL)
                outputs=outputs.reshape(1,self.NB_LABEL)

                if self.opt.norm_method == "standardization" or self.opt.norm_method == "minmax":
                    outputs = self.scaler.inverse_transform(outputs)
                    labels = self.scaler.inverse_transform(labels)
                output[i] = outputs
                label[i] = labels
                IDs[i] = ID[0]
            name_out = "./result" + str(epoch) + ".pkl"
            mse = test_loss/test_total
            with open(os.path.join(self.save_fold,name_out),"wb") as f:
                pickle.dump({"output":output,"label":label,"ID":IDs},f)
            #with open(os.path.join(self.save_fold,name_lab),"wb") as f:
                #pickle.dump(label,f)
           
        print(' Test_loss: {}'.format(test_loss/test_total))
        return mse
