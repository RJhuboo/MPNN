from collections import OrderedDict
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F

 ## Neural Network for regression ##
class NeuralNet(nn.Module):
    def __init__(self,n1,n2,n3,out_channels):
        super().__init__()
        self.fc1 = nn.Linear((64*64*64)+(64*64),n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        self.fc4 = nn.Linear(n3,out_channels)
    def forward(self,mask,x):
        mask = torch.flatten(mask,1)
        x = torch.flatten(x,1)
        x = torch.cat((x,mask),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
 ## Distance map ##
class DMNet(nn.Module):
    def __init__(self):
        super(DMNet, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_transpose1(x)
        x = self.dropout(x)
        x= self.conv2(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = self.relu(x)
        return x

class SkelNet(nn.Module):
    def __init__(self):
        super(SkelNet, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_transpose1(x)
        x = self.dropout(x)
        x= self.conv2(x)
        x = self.relu(x)
        x = self.conv_transpose2(x)
        x = self.sigmoid(x)
        return x

 
## 3 CNN model ##
class ConvNet(nn.Module):
    def __init__(self,in_channel,features,out_channels,n1=240,n2=120,n3=60,k1=3,k2=3,k3=3):
        super(ConvNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(in_channel,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout2d(0.25)
        self.neural = NeuralNet(n1,n2,n3,out_channels)
        self.SkelNet = SkelNet()
        self.DMNet = DMNet()
    def forward(self, mask,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x= self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        measure = self.neural(mask,x)
        skel = self.SkelNet(x)
        Dist = self.DMNet(x)
        return measure,skel,Dist
       
