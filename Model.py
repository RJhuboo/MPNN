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
        #self.fc1 = nn.Linear(64*64*64,n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        #self.fc5 = nn.Linear(n3,20)
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

## CNN model ##
class ConvNet(nn.Module):
    def __init__(self,in_channel,features,out_channels,n1=240,n2=120,n3=60,k1=3,k2=3,k3=3):
        super(ConvNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(in_channel,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # initialize NN layers
        #self.fc1 = nn.Linear(64**3,n1)
        #self.fc2 = nn.Linear(n1,n2)
        #self.fc3 = nn.Linear(n2,14)
        self.dropout = nn.Dropout2d(0.25)
        self.neural = NeuralNet(n1,n2,n3,out_channels)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self, mask,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x= self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.neural(mask,x)
        #x = torch.flatten(x,1)
        return x 
       
## Multitasking ##
class MultiNet(nn.Module):
    def __init__(self, features,out_channels,n1=240,n2=120,n3=60,k1=3,k2=3,k3=3):
        super(MultiNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(1,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # initialize NN layers
        self.neural_p1 = NeuralNet(n1,n2,n3,1)
        self.neural_p2 = NeuralNet(n1,n2,n3,1)
        self.neural_p3 = NeuralNet(n1,n2,n3,1)
        self.neural_p4 = NeuralNet(n1,n2,n3,1)
        self.neural_p5 = NeuralNet(n1,n2,n3,1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        p1 = self.neural_p1(x)
        p2 = self.neural_p2(x)
        p3 = self.neural_p3(x)
        p4 = self.neural_p4(x)
        p5 = self.neural_p5(x)
        return [p1,p2,p3,p4,p5]
    
## UNET model ##
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1,nb_label=14, n1=240,n2=120,n3=60, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")


        #self.conv = nn.Conv2d(
        #    in_channels=features, out_channels=out_channels, kernel_size=1
        #)

        self.conv = nn.Conv2d(in_channels=features, out_channels=features*3, kernel_size=3,stride=1,padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=features*3, out_channels=features*2,kernel_size=3,stride=1,padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(features*2,64,kernel_size=3,stride=1,padding=1)
        self.pool7 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.neural = NeuralNet(n1,n2,n3,nb_label)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        x = self.pool5(F.relu(self.conv(dec1)))
        x = self.pool6(F.relu(self.conv2(x)))
        x = self.pool7(F.relu(self.conv3(x)))
        return self.neural(x)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
