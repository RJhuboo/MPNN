import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from torch.optim import Adam, SGD
from sklearn.metrics import r2_score
from skimage import io,transform
from torchvision import transforms, utils
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import optuna

NB_DATA = 3991
NB_LABEL = 5
PERCENTAGE_TEST = 20
RESIZE_IMAGE = 512

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, opt, indices,transform=None):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
        self.indices = indices
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0]))
        image = io.imread(img_name) # Loading Image
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        labels = self.labels.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        return {"image":image,"label":labels}
class NeuralNet(nn.Module):
    def __init__(self,n1,n2,n3,out_channels):
        super().__init__()
        self.fc1 = nn.Linear(64*64*64,n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        #self.fc5 = nn.Linear(n3,20)
        self.fc4 = nn.Linear(n3,out_channels)
    def forward(self,x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc5(x))
        x = self.fc4(x)
        return x
class ConvNet(nn.Module):
    def __init__(self,features,out_channels,n1=240,n2=120,n3=60,k1=3,k2=3,k3=3):
        super(ConvNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(1,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # initialize NN layers
        #self.fc1 = nn.Linear(64**3,n1)
        #self.fc2 = nn.Linear(n1,n2)
        #self.fc3 = nn.Linear(n2,14)
        self.neural = NeuralNet(n1,n2,n3,out_channels)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.neural(x)
        #x = torch.flatten(x,1)
        return x 
def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()
def train(model,trainloader, optimizer, epoch ,steps_per_epochs=20):
    model.train()
    print("starting training")
    print("----------------")
    train_loss = 0.0
    train_total = 0
    running_loss = 0.0
    r2_s = 0
    mse_score = 0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data['image'], data['label']
        # reshape
        inputs = inputs.reshape(inputs.size(0),1,512,512)
        labels = labels.reshape(labels.size(0),NB_LABEL)
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward backward and optimization
        outputs = model(inputs)
        print(outputs.shape)
        print(labels.shape)
        Loss = MSELoss()
        loss = Loss(outputs,labels)
        loss.backward()
        optimizer.step()
        # statistics
        train_loss += loss.item()
        running_loss += loss.item()
        train_total += labels.size(0)
        outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        labels, outputs = np.array(labels), np.array(outputs)
        labels, outputs = labels.reshape(NB_LABEL,len(inputs)), outputs.reshape(NB_LABEL,len(inputs))
        r2 = r2_score(labels,outputs)
        r2_s += r2
        #Loss = MSELoss()
        mse_score += loss
        if i % opt['batch_size'] == opt['batch_size']-1:
            print('[%d %5d], loss: %.3f' %
                  (epoch + 1, i+1, running_loss/opt['batch_size']))
            running_loss = 0.0
    # displaying results
    mse = mse_score / i
    r2_s = r2_s/i
    print('Epoch [{}], Loss: {}, R square: {}'.format(epoch+1, train_loss/train_total, r2_s), end='')
    print('Finished Training')

    return mse

def test(model,testloader,epoch):
    model.eval()

    test_loss = 0
    test_total = 0
    r2_s = 0
    mse_score = 0.0
    output = {}
    label = {}
    # Loading Checkpoint
    if opt['mode'] == "Test":
        check_name = "BPNN_checkpoint_" + str(epoch) + ".pth"
        model.load_state_dict(torch.load(os.path.join(opt['checkpoint_path'],check_name)))
    # Testing
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data['image'],data['label']
            # reshape
            inputs = inputs.reshape(1,1,512,512)
            labels = labels.reshape(1,NB_LABEL)
            inputs, labels = inputs.to(device),labels.to(device)
            # loss
            outputs = model(inputs)
            Loss = MSELoss()
            test_loss += Loss(outputs,labels)
            test_total += labels.size(0)
            # statistics
            outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
            labels, outputs = np.array(labels), np.array(outputs)
            labels, outputs = labels.reshape(NB_LABEL,1), outputs.reshape(NB_LABEL,1)
            r2 = r2_score(labels,outputs)
            r2_s += r2
            #Loss = MSELoss()
            mse_score += loss

            outputs,labels=outputs.reshape(1,NB_LABEL), labels.reshape(1,NB_LABEL)
            output[i] = outputs
            label[i] = labels
        name_out = "./output" + str(epoch) + ".txt"
        name_lab = "./label" + str(epoch) + ".txt"
        mse = mse_score/i


    r2_s = r2_s/i
    print(' Test_loss: {}, Test_R_square: {}'.format(test_loss/test_total, r2_s))
    return mse

def objective(trial):
    # Create the folder where to save results and checkpoints
    mse_train = []
    mse_test = []
    opt = {'label_dir' : "./Label_5p.csv",
           'image_dir' : "./data/ROI_trab",
           'train_cross' : "./cross_output.pkl",
           'batch_size' : 8,
           'model' : "ConvNet",
           'nof' : 8,
           'lr': trial.suggest_loguniform('lr',1e-3,1e-2),
           'nb_epochs' : 5,
           'checkpoint_path' : "./",
           'mode': "Train",
           'cross_val' : False,
           'k_fold' : 5,
           'n1' : 240,
           'n2' : 120,
           'n3' : 60,
           'nb_workers' : 0,
           'norm_method':"standardization",
           'optimizer' :  trial.suggest_categorical('optimizer',[SGD, Adam])

          }
    
    # defining data
    index = range(NB_DATA)
    split = train_test_split(index,test_size = 0.2,random_state=1)
    datasets = Datasets(csv_file = opt['label_dir'], image_dir = opt['image_dir'], opt=opt, indices = split[0]) # Create dataset
    print("start training")
    trainloader = DataLoader(datasets, batch_size = opt['batch_size'], sampler = split[0], num_workers = opt['nb_workers'] )
    testloader =DataLoader(datasets, batch_size = 1, sampler = split[1], num_workers = opt['nb_workers'] )
    model = ConvNet(features =opt['nof'],out_channels=NB_LABEL,k1 = 3,k2 = 3,k3= 3).to(device)
    model.apply(reset_weights)
    optimizer = opt['optimizer'](model.parameters(), lr=opt['lr'])
    for epoch in range(opt['nb_epochs']):
        mse_train.append(train(model = model, trainloader = trainloader,optimizer = optimizer,epoch = epoch))
        mse_test.append(test(model=model,testloader=testloader,epoch=epoch))
    return max(mse_test)

''''''''''''''''''''' MAIN '''''''''''''''''''''''

if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
    
study.optimize(objective,n_trials=20)
joblid.dump(study,'./train_optuna.pkl')


