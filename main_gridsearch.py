import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn import MSELoss,L1Loss
from torch.optim import Adam, SGD
import torchvision.transforms.functional as TF
import random
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
import joblib
from math import isnan
import time
from sklearn.utils import shuffle

NB_DATA = 7100
NB_LABEL = 9
PERCENTAGE_TEST = 20
RESIZE_IMAGE = 512

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')

def normalization(csv_file,mode,indices):
    Data = pd.read_csv(csv_file)
    if mode == "standardization":
        scaler = preprocessing.StandardScaler()
    elif mode == "minmax":
        scaler = preprocessing.MinMaxScaler()
    scaler.fit(Data.iloc[indices,1:])
    return scaler

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, scaler, opt):
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.scaler=scaler
        self.mask_dir = mask_dir
        self.mask_use = True
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        mask_name = os.path.join(self.mask_dir, str(self.labels.iloc[idx,0][:-4] + ".bmp"))
        image = io.imread(img_name) # Loading Image
        if self.mask_use == True:
            mask = io.imread(mask_name)
            mask = transform.rescale(mask, 1/8, anti_aliasing=False)
            mask = mask / 255.0 # Normalizing [0;1]
            mask = mask.astype('float32') # Converting images to float32
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32
        else:
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32 
        lab = self.scaler.transform(self.labels.iloc[:,1:])
        lab = pd.DataFrame(lab)
        lab.insert(0,"File name", self.labels.iloc[:,0], True)
        lab.columns = self.labels.columns
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        p = random.random()
        rot = random.randint(-45,45)
        transform_list = []
        image,mask=TF.to_pil_image(image),TF.to_pil_image(mask)
        image,mask=TF.rotate(image,rot),TF.rotate(mask,rot)
        if p<0.3:
            image,mask=TF.vflip(image),TF.vflip(mask)
        p = random.random()
        if p<0.3:
            image,mask=TF.hflip(image),TF.hflip(mask)
        p = random.random()
        if p>0.2:
            image,mask=TF.affine(image,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(mask,angle=0,translate=(0.1,0.1),shear=0,scale=1)
        image,mask=TF.to_tensor(image),TF.to_tensor(mask)
        return {'image': image,'mask':mask, 'label': labels}
    
class NeuralNet(nn.Module):
    def __init__(self,activation,n1,n2,n3,out_channels):
        super().__init__()
        self.fc1 = nn.Linear((64*64*64)+(64*64),n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        self.fc4 = nn.Linear(n3,out_channels)
        self.activation = activation
    def forward(self,x,mask):
        x = torch.flatten(x,1)
        mask = torch.flatten(mask,1)
        x = torch.cat((x,mask),1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
class ConvNet(nn.Module):
    def __init__(self,activation, features,out_channels,n1=240,n2=120,n3=60,k1=3,k2=3,k3=3):
        super(ConvNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(1,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.activation = activation
        # initialize NN layers
        self.neural = NeuralNet(activation,n1,n2,n3,out_channels)
        # Dropout
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x, mask):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.activation(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.activation(self.conv3(x)))
        x = self.neural(x,mask)
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

def train(model,trainloader, optimizer, epoch , opt, steps_per_epochs=20):
    model.train()
    print("starting training")
    print("----------------")
    train_loss = 0.0
    train_total = 0
    running_loss = 0.0
    r2_s = 0
    mse_score = 0.0
    Loss= L1Loss()
    for i, data in enumerate(trainloader,0):
        inputs, masks, labels = data['image'],data['mask'], data['label']
        # reshape
        inputs = inputs.reshape(inputs.size(0),1,RESIZE_IMAGE,RESIZE_IMAGE)
        labels = labels.reshape(labels.size(0),NB_LABEL)
        masks = masks.reshape(masks.size(0),1,64,64)
        inputs, labels, masks= inputs.to(device), labels.to(device), masks.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward backward and optimization
        outputs = model(inputs,masks)
        loss = Loss(outputs,labels)
        if isnan(loss) == True:
            print(outputs)
            print(labels)

        loss.backward()
        optimizer.step()
        # statistics
        train_loss += loss.item()
        running_loss += loss.item()
        train_total += 1
        #outputs, labels = outputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        #labels, outputs = np.array(labels), np.array(outputs)
        #labels, outputs = labels.reshape(NB_LABEL,len(inputs)), outputs.reshape(NB_LABEL,len(inputs))
        #Loss = MSELoss()
        if i % opt['batch_size'] == opt['batch_size']-1:
            print('[%d %5d], loss: %.3f' %
                  (epoch + 1, i+1, running_loss/opt['batch_size']))
            running_loss = 0.0
        
    # displaying results
    mse = train_loss/train_total   
    print('Epoch [{}], Loss: {}'.format(epoch+1, train_loss/train_total), end='')
    print('Finished Training')

    return mse

def test(model,testloader,epoch,opt):
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
    Loss=L1Loss()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels, masks = data['image'],data['label'], data['mask']
            # reshape
            inputs = inputs.reshape(1,1,RESIZE_IMAGE,RESIZE_IMAGE)
            labels = labels.reshape(1,NB_LABEL)
            masks = masks.reshape(1,1,64,64)
            inputs, labels, masks = inputs.to(device),labels.to(device),masks.to(device)
            # loss
            outputs = model(inputs,masks)
            loss = Loss(outputs,labels)
            test_loss += loss.item()
            test_total += 1
            # statistics

            #outputs,labels=outputs.reshape(1,NB_LABEL), labels.reshape(1,NB_LABEL)
            #output[i] = outputs
            #label[i] = labels
        #name_out = "./output" + str(epoch) + ".txt"
        #name_lab = "./label" + str(epoch) + ".txt"



    print(' Test_loss: {}'.format(test_loss/test_total))
    return (test_loss/test_total)


def objective(trial):
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/cross_9p_augment"+str(i)) == False:
            save_folder = "./result/cross_9p_augment"+str(i)
            os.mkdir(save_folder)
            break
    # Create the folder where to save results and checkpoints
    opt = {'label_dir' : "./Train_Label_9p_augment.csv",
           'image_dir' : "./Train_segmented_filtered",
           'mask_dir' : "./Train_trab_mask",
           'batch_size' : trial.suggest_int('batch_size',8,24,step=8),
           #'batch_size': 24,
           'model' : "ConvNet",
           'nof' : trial.suggest_int('nof',8,64),
           #'nof':36,
           'lr': trial.suggest_loguniform('lr',1e-7,1e-3),
           #'lr':0.00006,
           'nb_epochs' : 250,
           'checkpoint_path' : "./",
           'mode': "Train",
           'cross_val' : False,
           'k_fold' : 1,
           #'n1': 135,
           #'n2':146,
           #'n3':131,
           'n1' : trial.suggest_int('n1', 90,190),
           'n2' : trial.suggest_int('n2',100,200),
           'n3' : trial.suggest_int('n3',100,190),
           'nb_workers' : 6,
           #'norm_method': trial.suggest_categorical('norm_method',["standardization","minmax"]),
           'norm_method': "standardization",
           'optimizer' :  trial.suggest_categorical("optimizer",[Adam]),
           #'optimizer': Adam,
           'activation' : trial.suggest_categorical("activation", [F.relu]),                                         
          }
    
    # defining data
    mse_train = []
    index = range(NB_DATA)
    split = train_test_split(index,train_size=6100,test_size=1000,shuffle=False)
    #kf = KFold(n_splits = opt['k_fold'], shuffle=False)
    print("start training")
    mse_total = np.zeros(opt['nb_epochs'])

    train_index=split[0]
    test_index=split[1]
    mse_test = []
    scaler = normalization(opt['label_dir'],opt['norm_method'],train_index)
    datasets = Datasets(csv_file = opt['label_dir'], image_dir = opt['image_dir'], mask_dir = opt['mask_dir'], opt=opt, scaler=scaler)
    trainloader = DataLoader(datasets, batch_size = opt['batch_size'], sampler = shuffle(train_index), num_workers = opt['nb_workers'])
    testloader =DataLoader(datasets, batch_size = 1, sampler = shuffle(test_index), num_workers = opt['nb_workers'])
    torch.manual_seed(5)
    model = ConvNet(activation = opt['activation'],features =opt['nof'],out_channels=NB_LABEL,n1=opt['n1'],n2=opt['n2'],n3=opt['n3'],k1 = 3,k2 = 3,k3= 3).to(device)
    #model.apply(reset_weights)
    optimizer = opt['optimizer'](model.parameters(), lr=opt['lr'])
    for epoch in range(opt['nb_epochs']):
        mse_train.append(train(model = model, trainloader = trainloader,optimizer = optimizer,epoch = epoch,opt=opt))
        mse_test.append(test(model=model, testloader=testloader, epoch=epoch, opt=opt))
    #mse_total = mse_total + np.array(mse_test)
    #print("mse train size :",len(mse_train))
    #mse_mean = mse_total / opt['k_fold']
    print("min mse test :", np.min(mse_test))
    i_min = np.where(mse_test == np.min(mse_test))
    print('best epoch :', i_min[0][0]+1)
    result_display = {"train mse":mse_train,"val mse":mse_test,"best epoch":i_min[0][0]+1}
    with open(os.path.join(save_folder,"training_info.pkl"),"wb") as f:
        pickle.dump(result_display,f)
    return np.min(mse_test)

''''''''''''''''''''' MAIN '''''''''''''''''''''''

if torch.cuda.is_available():
    device = "cuda:0"
    print("running on gpu")
else:  
    device = "cpu"
    print("running on cpu")
    
study.optimize(objective,n_trials=15)
with open("./cross_9p_augment.pkl","wb") as f:
    pickle.dump(study,f)
