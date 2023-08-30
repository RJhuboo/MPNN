import torch
import os
import numpy as np
from torch.optim import Adam, SGD
from torch.nn import L1Loss
import pickle
from matplotlib import pyplot as plt
import torchvision

def MSE(y_predicted,y,batch_size):
    squared_error = abs((y_predicted.cpu().detach().numpy() - y.cpu().detach().numpy()))
    sum_squared_error = np.sum(np.array(squared_error))
    mse = sum_squared_error / batch_size
    return mse

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
        #self.model.train()
        self.model_TF.train()
        self.model.eval()
        print("starting training")
        print("----------------")
        train_loss = 0.0
        train_total = 0
        running_loss = 0.0

        L1_loss_train=np.zeros((len(trainloader),self.NB_LABEL))
        for i, data in enumerate(trainloader,0):
            inputs, masks, labels, imname = data['image'], data['mask'], data['label'], data['ID']
            
            # reshape
            inputs = inputs.reshape(inputs.size(0),self.opt.in_channel,512,512)
            labels = labels.reshape(labels.size(0),self.NB_LABEL)
            masks = masks.reshape(masks.size(0),1,64,64)
            inputs, labels, masks= inputs.to(self.device), labels.to(self.device), masks.to(self.device)

            #torchvision.utils.save_image(inputs,'./save_image/input_'+imname[0])
            #torchvision.utils.save_image(masks,'./save_image/mask_'+imname[0])
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward backward and optimization
            outputs = self.model(masks,inputs)
            #outputs = self.model(inputs)
            if self.opt.model == "MultiNet":
                loss1 = self.criterion(outputs[0],torch.reshape(labels[:,0],[len(outputs[0]),1]))
                loss2 = self.criterion(outputs[1],torch.reshape(labels[:,1],[len(outputs[1]),1]))
                loss3 = self.criterion(outputs[2],torch.reshape(labels[:,2],[len(outputs[2]),1]))
                loss4 = self.criterion(outputs[3],torch.reshape(labels[:,3],[len(outputs[3]),1]))
                loss5 = self.criterion(outputs[4],torch.reshape(labels[:,4],[len(outputs[4]),1]))
                loss = (self.opt.alpha1*loss1) + (self.opt.alpha2*loss2) + (self.opt.alpha3*loss3) + (self.opt.alpha4*loss4) + (self.opt.alpha5*loss5)
            else:
                loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            for nb_lab in range(self.NB_LABEL): 
                L1_loss_train[i,nb_lab] = MSE(labels[:,nb_lab],outputs[:,nb_lab],24)
            
            # statistics
            train_loss += loss.item()
            running_loss += loss.item()
            train_total += 1
            #labels, outputs = labels.reshape(self.NB_LABEL,len(inputs)), outputs.reshape(self.NB_LABEL,len(inputs))
            if i % self.opt.batch_size == self.opt.batch_size-1:
                print('[%d %5d], loss: %.3f' %
                      (epoch + 1, i+1, running_loss/self.opt.batch_size))
                running_loss = 0.0
                #print("output",outputs[:8])
                #print("label",labels[:8])
        # displaying results
        mse = train_loss / train_total
        print('Epoch [{}], Loss: {}'.format(epoch+1, train_loss/train_total), end='')
        print('Finished Training')
        
        # saving trained model
        #if epoch > 1:
        #    print("---- saving model ----")
        #    check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        #    torch.save(self.model.state_dict(),os.path.join(self.opt.checkpoint_path,check_name))
        return mse, np.mean(L1_loss_train,axis=0)

    def test(self,testloader,epoch,writer):

        test_loss = 0
        test_total = 0
        mse_score = 0.0
        output = []
        label = []
        IDs = {}
        # Loading Checkpoint
        if self.opt.mode == "Test":
            check_name = "BPNN_checkpoint_lrhr.pth" #+ str(epoch) + ".pth"
            self.model.load_state_dict(torch.load(os.path.join(self.opt.checkpoint_path,check_name)))
        
        self.model.eval()
        self.model_TF.eval()
        L1_loss_test=np.zeros((len(testloader),self.NB_LABEL))
        # Testing
        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, masks, labels, ID = data['image'],data['mask'],data['label'],data['ID']
                # reshape
                inputs = inputs.reshape(1,self.opt.in_channel,512,512)
                labels = labels.reshape(1,self.NB_LABEL)
                masks = masks.reshape(1,1,64,64)
                inputs, labels, masks= inputs.to(self.device),labels.to(self.device), masks.to(self.device)
       
                # loss
                outputs = self.model(masks,inputs)
                #if 1 in outputs.clamp(-1,1) or -1 in outputs.clamp(-1,1):
                #outputs = self.model(inputs)
                if self.opt.model == "MultiNet":
                    loss1 = self.criterion(outputs[0],torch.reshape(labels[:,0],[1,1]))
                    loss2 = self.criterion(outputs[1],torch.reshape(labels[:,1],[1,1]))
                    loss3 = self.criterion(outputs[2],torch.reshape(labels[:,2],[1,1]))
                    loss4 = self.criterion(outputs[3],torch.reshape(labels[:,3],[1,1]))
                    loss5 = self.criterion(outputs[4],torch.reshape(labels[:,4],[1,1]))
                    loss = (self.opt.alpha1*loss1) + (self.opt.alpha2*loss2) + (self.opt.alpha3*loss3) + (self.opt.alpha4*loss4) + (self.opt.alpha5*loss5)
                else:
                    loss = self.criterion(outputs,labels)
                test_loss += loss.item()
                test_total += 1
                for nb_lab in range(self.NB_LABEL):
                    L1_loss_test[i,nb_lab] = MSE(labels[0,nb_lab],outputs[0,nb_lab],1)
                # statistics
                if self.opt.model == "MultiNet":
                    labels = labels.cpu().detach().numpy()
                else:
                    labels, outputs = labels.cpu().detach().numpy(), outputs.cpu().detach().numpy()
                #torchvision.utils.save_image(inputs, "./save_image/"+ ID[0])
                #torchvision.utils.save_image(masks, "./save_image/mask_"+ ID[0])
                labels, outputs = np.array(labels), np.array(outputs)
                #labels, outputs = labels.reshape(self.NB_LABEL,1), outputs.reshape(self.NB_LABEL,1)
                labels=labels.reshape(1,self.NB_LABEL)
                outputs=outputs.reshape(1,self.NB_LABEL) 
                #print("labels",labels)
                if self.scaler is not None:
                    outputs = self.scaler.inverse_transform(outputs)
                    labels = self.scaler.inverse_transform(labels)
                output.append(outputs)
                label.append(labels)
                IDs[i] = ID[0]
                
            #name_out = "./result" + str(epoch) + ".pkl"
            mse = test_loss/test_total
            size_label = len(label)
            label = np.array(label)
            output = np.array(output)
            output, label = output.reshape((size_label,7)), label.reshape((size_label,7))
            print(np.shape(label))
            for i in range(np.shape(label)[1]):
                fig, ax = plt.subplots()
                ax.scatter(label[:,i],output[:,i], label="slice")
                ax.plot(label[:,i],label[:,i])
                #plt.xlabel("label")
                #plt.ylabel("output")
                writer.add_figure(str(epoch) + '/' + str(i),fig)
            #with open(os.path.join(self.save_fold,name_out),"wb") as f:
            #    pickle.dump({"output":output,"label":label,"ID":IDs},f)
            #with open(os.path.join(self.save_fold,name_lab),"wb") as f:
                #pickle.dump(label,f)
        #print(outputs)
        #print(labels)
        print(' Test_loss: {}'.format(test_loss/test_total))
        return mse, np.mean(L1_loss_test,axis=0)
    
