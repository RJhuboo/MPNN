import torch
import os
import numpy as np
from torch.optim import Adam, SGD
from torch.nn import MSELoss,L1Loss
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt

def L1(y_predicted,y,batch_size):
    sub = abs((y_predicted.cpu().detach().numpy() - y.cpu().detach().numpy()))
    sum_sub = np.sum(np.array(sub))
    l1 = sum_sub / batch_size
    return l1

class Trainer():
    def __init__(self,opt,my_model,device,save_fold,scaler):
        self.scaler = scaler
        self.save_fold = save_fold
        self.device = device
        self.opt = opt
        self.model = my_model
        self.NB_LABEL = opt.NB_LABEL
        if opt.optim == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.opt.lr)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.opt.lr)
        self.criterion = L1Loss()
        
    def train(self, trainloader, epoch ,steps_per_epochs=20):
        self.model.train()
        print("starting training")
        print("----------------")
        
        train_loss = 0.0
        train_total = 0
        running_loss = 0.0
        L1_loss_train=np.zeros((len(trainloader),self.NB_LABEL))
        
        for i, data in enumerate(trainloader,0):
            inputs, masks, labels, imname = data['image'], data['mask'], data['label'], data['ID']
            inputs = inputs.reshape(inputs.size(0),self.opt.in_channel,512,512)
            labels = labels.reshape(labels.size(0),self.NB_LABEL)
            masks = masks.reshape(masks.size(0),1,64,64)
            inputs, labels, masks= inputs.to(self.device), labels.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(masks,inputs)

            loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()

            # Measure a L1 loss for each biological parameter 
            for nb_lab in range(self.NB_LABEL): 
                L1_loss_train[i,nb_lab] = MSE(labels[:,nb_lab],outputs[:,nb_lab],24)
            
            # Performance evaluation
            train_loss += loss.item()
            running_loss += loss.item()
            train_total += 1      
            if i % self.opt.batch_size == self.opt.batch_size-1:
                print('[%d %5d], loss: %.3f' %
                      (epoch + 1, i+1, running_loss/self.opt.batch_size))
                running_loss = 0.0

        # displaying results
        perf_loss = train_loss / train_total
        print('Epoch [{}], Loss: {}'.format(epoch+1, train_loss/train_total), end='')
        print('Finished Training')
        
        # saving trained model
        if epoch > 50:
            print("---- saving model ----")
            check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
            torch.save(self.model.state_dict(),os.path.join(self.opt.checkpoint_path,check_name))
        return perf_loss, np.mean(L1_loss_train,axis=0)

    def test(self,testloader,writer,epoch):

        test_loss = 0
        test_total = 0
        output = []
        label = []
        IDs = {}
        
        # Loading Checkpoint
        if self.opt.mode == "Test":
            check_name = "BPNN_checkpoint_lrhr.pth" 
            self.model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        
        self.model.eval()
        L1_loss_test=np.zeros((len(testloader),self.NB_LABEL))

        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, masks, labels, ID = data['image'],data['mask'],data['label'],data['ID']
                inputs = inputs.reshape(1,self.opt.in_channel,512,512)
                labels = labels.reshape(1,self.NB_LABEL)
                masks = masks.reshape(1,1,64,64)
                inputs, labels, masks= inputs.to(self.device),labels.to(self.device), masks.to(self.device)

                outputs = self.model(masks,inputs)
                loss = self.criterion(outputs,labels)
                test_loss += loss.item()
                test_total += 1

                # L1 loss per biological parameter
                for nb_lab in range(self.NB_LABEL):
                    L1_loss_test[i,nb_lab] = MSE(labels[0,nb_lab],outputs[0,nb_lab],1)
                
                labels, outputs = labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()
                labels, outputs = np.array(labels), np.array(outputs)
                labels=labels.reshape(1,self.NB_LABEL)
                outputs=outputs.reshape(1,self.NB_LABEL)

                # Inverse normalization to get real parameter value
                outputs = self.scaler.inverse_transform(outputs)
                labels = self.scaler.inverse_transform(labels)
                
                output.append(outputs)
                label.append(labels)
                IDs[i] = ID[0]
            
            perf_loss = test_loss/test_total

            # Plot on tensorboard
            size_label = len(label)
            label = np.array(label)
            output = np.array(output)
            output, label = output.reshape((size_label,7)), label.reshape((size_label,7))
            for i in range(np.shape(label)[1]):
                fig, ax = plt.subplots()
                ax.scatter(label[:,i],output[:,i], label="slice")
                ax.plot(label[:,i],label[:,i])
                writer.add_figure(str(epoch)+"/"+str(i),fig)

        print(' Test_loss: {}'.format(test_loss/test_total))
        return perf_loss, np.mean(L1_loss_test,axis=0)
    

