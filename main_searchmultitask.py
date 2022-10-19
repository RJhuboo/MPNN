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
import joblib
from math import isnan
import time
NB_DATA = 4073
NB_LABEL = 6
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
        img_name = os.path.join(self.image_dir, str(self.labels.iloc[idx,0][:-4] + ".png"))
        image = io.imread(img_name) # Loading Image
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32
        if self.opt['norm_method']== "L2":
            lab = preprocessing.normalize(self.labels.iloc[:,1:],axis=0)
        elif self.opt['norm_method'] == "L1":
            lab = preprocessing.normalize(self.labels.iloc[:,1:],norm='l1',axis=0)
        elif self.opt['norm_method'] == "minmax":
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(self.labels.iloc[self.indices,1:])
            lab = scaler.transform(self.labels.iloc[:,1:])
        elif self.opt['norm_method'] == "standardization":
            scaler = preprocessing.StandardScaler()
            scaler.fit(self.labels.iloc[self.indices,1:])
            lab = scaler.transform(self.labels.iloc[:,1:])
        lab = pd.DataFrame(lab)
        lab.insert(0,"File name", self.labels.iloc[:,0], True)
        lab.columns = self.labels.columns
        labels = lab.iloc[idx,1:] # Takes all corresponding labels
        labels = np.array([labels]) 
        labels = labels.astype('float32')
        if self.transform:
            sample = self.transform(sample)
        return {'image': image, 'label': labels}
    
class FFNN(nn.Module):
    """Simple FF network with multiple outputs.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        n_outputs,
        dropout_rate=.1,
    ):
        """
        :param input_size: input size
        :param hidden_size: common hidden size for all layers
        :param n_hidden: number of hidden layers
        :param n_outputs: number of outputs
        :param dropout_rate: dropout rate
        """
        super().__init__()
        assert 0 <= dropout_rate < 1
        self.input_size = input_size

        h_sizes = [self.input_size] + [hidden_size for _ in range(n_hidden)] + [n_outputs]

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(
                nn.Linear(
                    h_sizes[k],
                    h_sizes[k + 1]
                )
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        for layer in self.hidden[:-1]:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        return self.hidden[-1](x)
    
class TaskIndependentNets(nn.Module):
    """Independent FFNN for each task
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs,
            dropout_rate=.1,
    ):

        super().__init__()

        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(
                FFNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_hidden=n_hidden,
                    n_outputs=1,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, x):
 
        return torch.cat(
            tuple(task_model(x) for task_model in self.task_nets),
            dim=1
        )
class HardSharing(nn.Module):
    """FFNN with hard parameter sharing
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        n_outputs,
        n_task_specific_layers=0,
        task_specific_hidden_size=None,
        dropout_rate=.1,
    ):

        super().__init__()
        if task_specific_hidden_size is None:
            task_specific_hidden_size = hidden_size

        self.model = nn.Sequential()

        self.model.add_module(
            'hard_sharing',
            FFNN(
                input_size=input_size,
                hidden_size=hidden_size,
                n_hidden=n_hidden,
                n_outputs=hidden_size,
                dropout_rate=dropout_rate
            )
        )

        if n_task_specific_layers > 0:
            # if n_task_specific_layers == 0 than the task specific mapping is linear and
            # constructed as the product of last layer is the 'hard_sharing' and the linear layer
            # in 'task_specific', with no activation or dropout
            self.model.add_module('relu', nn.ReLU())
            self.model.add_module('dropout', nn.Dropout(p=dropout_rate))

        self.model.add_module(
            'task_specific',
            TaskIndependentLayers(
                input_size=hidden_size,
                hidden_size=task_specific_hidden_size,
                n_hidden=n_task_specific_layers,
                n_outputs=n_outputs,
                dropout_rate=dropout_rate
            )
        )
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        x = torch.flatten(x,1) 
        return self.model(x)
    
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
        print(outputs)
        print(labels)
        Loss = MSELoss()
        #labels = torch.transpose(labels,0,1)
        #loss_list = [Loss(outputs[k_task],torch.reshape(labels[k_task],[len(outputs[k_task]),1])) for k_task in range(NB_LABEL)]  
        #loss = sum(loss_list)
        loss = Loss(outputs,labels)
        #### verify non empty loss ####
        if isnan(loss) == True:
            print(outputs)
            print(labels)

        loss.backward()
        optimizer.step()
        # statistics
        train_loss += loss.item()
        running_loss += loss.item()
        train_total += 1

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
    mse_score = 0.0
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
            labels = torch.transpose(labels,0,1)
            #loss_list = [Loss(outputs[k_task],torch.reshape(labels[k_task],[1,1])) for k_task in range(NB_LABEL)]  
            #loss = sum(loss_list)
            loss = Loss(outputs,labels)
            test_loss += loss.item()
            test_total += 1
    print(' Test_loss: {}'.format(test_loss/test_total))
    return (test_loss/test_total)

def objective(trial):
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/multi_minmax"+str(i)) == False:
            save_folder = "./result/multi_minxmax"+str(i)
            os.mkdir(save_folder)
            break
    # Create the folder where to save results and checkpoints
    opt = {'label_dir' : "./Label_6p.csv",
           'image_dir' : "/gpfsstore/rech/tvs/uki75tv/MOUSE_BPNN/HR/Train_Label_trab",
           'train_cross' : "./cross_output.pkl",
           'batch_size' : trial.suggest_int('batch_size',8,24,step=8),
           'model' : "ConvNet",
           'nb_hidden_layer' : trial.suggest_int('nb_hidden_layer',2,50),
           'task_specific_hidden_size' : trial.suggest_int('task_specific_hidden_size',50,500),
           'hidden_size' : trial.suggest_int('hidden_size',100,10000),
           'lr': trial.suggest_loguniform('lr',1e-4,1e-2),
           'nb_epochs' : 70,
           'checkpoint_path' : "./",
           'mode': "Train",
           'cross_val' : False,
           'k_fold' : 4,
           'nb_workers' : 4,
           #'norm_method': trial.suggest_categorical('norm_method',["standardization","minmax"]),
           'norm_method': "minmax",
           'optimizer' :  trial.suggest_categorical("optimizer",[Adam, SGD]),                                                  
          }
    
    # defining data
    mse_train = []
    mse_total = np.zeros(opt['nb_epochs'])
    index = range(NB_DATA)
    split = train_test_split(index,test_size = 0.2,shuffle=False)
    kf = KFold(n_splits = opt['k_fold'], shuffle=False)
    
    print("start training")
    for train_index, test_index in kf.split(split[0]):
        mse_test = []
        if opt['norm_method'] == "standardization" or opt['norm_method'] == "minmax":
            scaler = normalization(opt['label_dir'],opt['norm_method'],train_index)
        else:
            scaler = None
        datasets = Datasets(csv_file = opt['label_dir'], image_dir = opt['image_dir'], opt=opt, indices = train_index) # Create dataset
        trainloader = DataLoader(datasets, batch_size = opt['batch_size'], sampler = train_index, num_workers = opt['nb_workers'])
        testloader =DataLoader(datasets, batch_size = 1, sampler = test_index, num_workers = opt['nb_workers'])
        model = HardSharing(input_size=512*512,hidden_size=opt['nb_hidden_layer'],n_hidden=opt['n_hidden'],n_outputs=NB_LABEL,task_specific_hidden_size=opt['task_specific_hidden_size']).to(device)
        optimizer = opt['optimizer'](model.parameters(), lr=opt['lr'])
        for epoch in range(opt['nb_epochs']):
            mse_train.append(train(model = model, trainloader = trainloader,optimizer = optimizer,epoch = epoch,opt=opt))
            mse_test.append(test(model = model, testloader = testloader, epoch = epoch,opt=opt))
        mse_total = np.array(mse_test) + mse_total    
    mse_mean = mse_total / opt['k_fold']
    i_min = np.where(mse_mean == np.min(mse_mean))
    print('best epoch :', i_min[0][0]+1)
    result_display = {"train mse":mse_train,"val mse":mse_mean, "best epoch":i_min[0][0]+1}
    with open(os.path.join(save_folder,"training_info.pkl"),"wb") as f:
        pickle.dump(result_display,f)
    return np.min(mse_mean)

''''''''''''''''''''' MAIN '''''''''''''''''''''''

if torch.cuda.is_available():
    device = "cuda:0"
    print("running on gpu")
else:  
    device = "cpu"
    print("running on cpu")
    
study.optimize(objective,n_trials=6)
with open("./cross_multitasking_minmax.pkl","wb") as f:
    pickle.dump(study,f)
