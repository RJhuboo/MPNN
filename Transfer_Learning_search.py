''' This code is a copy of main_gridsearch.py. It is modified to fit human dataset. The object is the tune the hyper-parameters for transfer-learning tasks.
    To work, this code requires the checkpoints of the trained model on mouse dataset.
    Then the current datasets is a human bone dataset.'''


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

NB_DATA = 500 # !!! Must be checked before running !!!
NB_LABEL = 7
PERCENTAGE_TEST = 20
RESIZE_IMAGE = 512

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')

# Normalization function for morphometry data
def normalization(csv_file,mode,indices):
    Data = pd.read_csv(csv_file)
    # Initialize scaler
    if mode == "standardization":
        scaler = preprocessing.StandardScaler()
    elif mode == "minmax":
        scaler = preprocessing.MinMaxScaler()
    # Compute mean and standard deviation
    scaler.fit(Data.iloc[indices,1:])
    return scaler

# Datasets constructor
class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir, scaler, opt):
        """ Initializes the datasets variables.

        Args:
            csv_file (_type_): Label csv file 
            image_dir (_type_): directory of the images
            mask_dir (_type_): directory of the masks
            scaler (_type_): normalization informations
            opt (_type_): Some options
        """
        self.opt = opt
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)
        self.labels = self.labels.drop(range(300,400))
        self.labels = self.labels.reset_index(drop=True)
        print(self.labels["File name"])
        self.scaler=scaler
        self.mask_dir = mask_dir
        self.mask_use = True # Tune for use of mask
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Find image path
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        mask_name = os.path.join(self.mask_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        
        # Read image and mask
        image = io.imread(img_name)
        if 'lr' in img_name: # If image is a low resolution image
            image = transform.rescale(image,2) # Rescaling the image to match size of high resolution image
            image = (image<0.5)*255 # Binarized the Image between 0 and 255
            mask_name = os.path.join(self.mask_dir,str(self.labels.iloc[idx,0]).replace("_lr.tif",".bmp")) # Find the corresponding mask file
        if self.mask_use == True:
            mask = io.imread(mask_name) # Read the mask
            mask = transform.rescale(mask, 1/8, anti_aliasing=False) # Rescaling the mask
            mask = mask / 255.0 # Normalizing [0;1]
            mask = mask.astype('float32') # Converting images to float32
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32
        else:
            image = image / 255.0 # Normalizing [0;1]
            image = image.astype('float32') # Converting images to float32 
        lab = self.scaler.transform(self.labels.iloc[:,1:]) # Apply the normalization to labels
        lab = pd.DataFrame(lab) # Converting labels to pandas dataframe
        lab.insert(0,"File name", self.labels.iloc[:,0], True) # Inset the name of images
        lab.columns = self.labels.columns # Take the columns names
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) # Converting labels to numpy array
        labels = labels.astype('float32') # Converting labels to float32
        
        # Image transformation for data augmentation
        p = random.random()
        rot = random.randint(-45,45)
        transform_list = []
        image,mask=TF.to_pil_image(image),TF.to_pil_image(mask)
        #image = torch.from_numpy(np.array(image, np.float32, copy=False))
        #mask = torch.from_numpy(np.array(mask, np.float32, copy=False))
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
    
# Dense neural network for regression task
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
        x = self.activation(self.fc4(x))
        return x
    
# Convolutional neural network for feature extraction task
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
        x = self.neural(x,mask) # Mask is used for the dense neural network
        return x

# A function for resetting model weights
def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

# Training function
def train(model,trainloader, optimizer, epoch , opt, steps_per_epochs=20):
    model.train() # Switch to train mode
    
    print("starting training")
    print("----------------")
    
    # Loss tracking and metric initialization
    train_loss = 0.0
    train_total = 0
    running_loss = 0.0

    # Loss initilization
    Loss= L1Loss() # Choose L1Loss or L2Loss
    
    # Training loop into all the datasets
    for i, data in enumerate(trainloader,0):
        # open data
        inputs, masks, labels = data['image'],data['mask'], data['label']
        # reshape the data to correct tensor shape
        inputs = inputs.reshape(inputs.size(0),1,RESIZE_IMAGE,RESIZE_IMAGE)
        labels = labels.reshape(labels.size(0),NB_LABEL)
        masks = masks.reshape(masks.size(0),1,64,64)
        # Send data to GPU
        inputs, labels, masks= inputs.to(device), labels.to(device), masks.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward backward
        outputs = model(inputs,masks)
        loss = Loss(outputs,labels)
        
        # Check for errors in training only use for debugging
        if isnan(loss) == True:
            print(outputs)
            print(labels)

        # backward and optimization
        loss.backward()
        optimizer.step()
        # statistics
        train_loss += loss.item()
        running_loss += loss.item()
        train_total += 1
        # Print statistics
        if i % opt['batch_size'] == opt['batch_size']-1:
            print('[%d %5d], loss: %.3f' %
                  (epoch + 1, i+1, running_loss/opt['batch_size']))
            running_loss = 0.0
        
    # displaying final results
    final_loss = train_loss/train_total   
    print('Epoch [{}], Loss: {}'.format(epoch+1, final_loss), end='')
    print('Finished Training')
    
    return final_loss # Returning the L1 loss

# Testing function: This more an evaluation step than a true testing step
def test(model,testloader,epoch,opt):
    # switch to evaluate mode
    model.eval()

    # Initilization of variables
    test_loss = 0
    test_total = 0
    
    # Loss initilization
    Loss=L1Loss()
    # Disable gradients computation
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels, masks = data['image'],data['label'], data['mask']
            # reshape
            inputs = inputs.reshape(1,1,RESIZE_IMAGE,RESIZE_IMAGE)
            labels = labels.reshape(1,NB_LABEL)
            masks = masks.reshape(1,1,64,64)
            inputs, labels, masks = inputs.to(device),labels.to(device),masks.to(device)
            outputs = model(inputs,masks)
            loss = Loss(outputs,labels)
            test_loss += loss.item()
            test_total += 1
            # Data storage if further analysis is required 

    print(' Test_loss: {}'.format(test_loss/test_total))
    return (test_loss/test_total)

# Objective function for optuna, allowing to track a metric evolution for each experiment.
def objective(trial):
    # Create the folder where to save results and checkpoints
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/cross_7p_transferlearning"+str(i)) == False:
            save_folder = "./result/cross_7p_transferlearning"+str(i)
            os.mkdir(save_folder)
            break
    
    # Options
    opt = {'label_dir' : "/gpfsstore/rech/tvs/uki75tv/Trab_Human.csv",
           'image_dir' : "/gpfsstore/rech/tvs/uki75tv/DATA_HUMAN/IMAGE",
           'mask_dir' : "/gpfsstore/rech/tvs/uki75tv/DATA_HUMAN/MASK",
           #'batch_size' : trial.suggest_int('batch_size',1,16,step=2),
           'batch_size': 1,
           'model' : "ConvNet",
           #'nof' : trial.suggest_int('nof',10,64),
           'layer_nb' : trial.suggest_int('layer_nb',1,3),
           'net_freeze': trial.suggest_categorical('net_freeze',[True,False]),
           'nof':64,
           'lr': trial.suggest_loguniform('lr',1e-4,1e-2),
           #'lr':0.00006,
           'nb_epochs' : 200,
           'checkpoint_path' : "./convnet_7p_lrhr/BPNN_checkpoint_449.pth",
           'mode': "Train",
           'cross_val' : False,
           'k_fold' : 5,
           'n1': 158,
           'n2':152,
           'n3':83,
           #'n1' : trial.suggest_int('n1', 80,200),
           #'n2' : trial.suggest_int('n2',90,200),
           #'n3' : trial.suggest_int('n3',80,190),
           'nb_workers' : 6,
           #'norm_method': trial.suggest_categorical('norm_method',["standardization","minmax"]),
           'norm_method': "standardization",
           'optimizer' :  trial.suggest_categorical("optimizer",[Adam]),
           #'optimizer': Adam,
           'activation' : trial.suggest_categorical("activation", [F.relu]),                                      
          }
    

    
    # Initilization of variables
    score_train_total = np.zeros(opt['nb_epochs'])
    score_total = np.zeros(opt['nb_epochs'])
    
    index = range(NB_DATA)
    indexes = []
    [indexes.append(i) for i in index]

    for k in range(opt["k_fold"]): # k-fold cross 
        
        # Create the fold vectors having full mouse data.
        train_index = []
        test_index = indexes[k*100:(1+k)*100]
        [train_index.append(i) for i in index if i not in test_index]
        #split = train_test_split(index,train_size=6100,test_size=1000,shuffle=False)
        #kf = KFold(n_splits = opt['k_fold'], shuffle=False)
        #train_index=split[0]
        #test_index=split[1]
        print("start training")
        
        # intilization of metric tracking at each fold
        score_test = []
        score_train = []
        # Scaler creation on the training dataset
        scaler = normalization(opt['label_dir'],opt['norm_method'],train_index)
        # Creation of the datasets  
        datasets = Datasets(csv_file = opt['label_dir'], image_dir = opt['image_dir'], mask_dir = opt['mask_dir'], opt=opt, scaler=scaler)
        trainloader = DataLoader(datasets, batch_size = opt['batch_size'], sampler = shuffle(train_index), num_workers = opt['nb_workers'])
        testloader =DataLoader(datasets, batch_size = 1, sampler = shuffle(test_index), num_workers = opt['nb_workers'])
        
        # Weight initilization        
        torch.manual_seed(5)
        # Model initilization
        model = ConvNet(activation = opt['activation'],features =opt['nof'],out_channels=NB_LABEL,n1=opt['n1'],n2=opt['n2'],n3=opt['n3'],k1 = 3,k2 = 3,k3= 3).to(device)
        
        # Loading Checkpoint of trained model.
        model.load_state_dict(torch.load(opt['checkpoint_path']))
        
        # Put model in multiple GpU mode
        if torch.cuda.device_count() >1:
            model = nn.DataParallel(model)
        #model.apply(reset_weights)
        # Optimizer initilization
        
        # Freeze the last two layers
        count = 0
        for name, param in model.named_parameters():
            if opt['net_freeze'] and count < opt['layer_nb']*2:
                param.requires_grad = False
            if opt['net_freeze'] == False and count < (opt['layer_nb']*2)+6:
                param.requires_grad = False
            count += 1

        # Verify the parameters
        print("Verify that freeze layer are:{}, and {}".format(opt['net_freeze'],opt['layer_nb']),)
        for name, param in model.named_parameters():
            print(f'{name}: requires_grad={param.requires_grad}')
    
        optimizer = opt['optimizer'](model.parameters(), lr=opt['lr'])
        
        # Training loop
        for epoch in range(opt['nb_epochs']):
            start = time.time() # Computational time 
            # Training
            score_train.append(train(model = model, trainloader = trainloader,optimizer = optimizer,epoch = epoch,opt=opt))
            end = time.time() # Computational time 
            print("temps :",start-end)
            # Testing
            score_test.append(test(model=model, testloader=testloader, epoch=epoch, opt=opt))
        # Store all folds scores
        score_total = score_total + np.array(score_test)
        score_train_total = score_train_total + np.array(score_train)
        
    # Compute mean score in function of number of folds
    score_mean = score_total / opt['k_fold']
    score_train_mean = score_train_total / opt['k_fold']
    # Display minimum score and best epoch
    print("min mse test :", np.min(score_mean))
    i_min = np.where(score_mean == np.min(score_mean))
    print('best epoch :', i_min[0][0]+1)
    result_display = {"train mse":score_train_mean,"val mse":score_mean,"best epoch":i_min[0][0]+1}
    # Save the results of the study to a pickle file
    with open(os.path.join(save_folder,"training_info.pkl"),"wb") as f:
            pickle.dump(result_display,f)
    return np.min(score_mean)


''''''''''''''''''''' MAIN '''''''''''''''''''''''
# Check if GPU is available and set the device
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# create a study on optuna for hyperparameter tuning
study.optimize(objective,n_trials=20) # n_trials is the number of experiments to run
# Save the results of the study to a pickle file
with open("./cross_7p_transferlearning_2.pkl","wb") as f:
    pickle.dump(study,f)
