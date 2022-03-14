import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,nof,NB_LABEL):
        super(Net,self).__init__()
        # initialize CNN layers
        self.conv1 = nn.Conv2d(1,nof,kernel_size = 3,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(nof,opt.nof*2, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(nof*2,opt.nof*4, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # initialize NN layers
        self.fc1 = nn.Linear(64*64*64,240)
        self.fc2 = nn.Linear(240,120)
        self.fc3 = nn.Linear(120,NB_LABEL)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
