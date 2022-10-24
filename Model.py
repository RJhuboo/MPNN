from collections import OrderedDict
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F

 ## Neural Network for regression ##
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

## 3 CNN model ##
class ConvNet(nn.Module):
    def __init__(self,features,k1=3,k2=3,k3=3):
        super(ConvNet,self).__init__()
        # initialize CNN layers 
        self.conv1 = nn.Conv2d(1,features,kernel_size = k1,stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(features,features*2, kernel_size = k2, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(features*2,64, kernel_size = k3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        # dropout
        # self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x,1)
        return x 
       
## Multitasking ##
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
    
class TaskIndependentLayers(nn.Module):
    """NN for MTL with hard parameter sharing
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
        
        self.model.add_module('convnet',ConvNet(features=40))
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
        
    #def _init_weights(self, module):
    #    if isinstance(module, nn.Linear):
    #        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    #        if module.bias is not None:
    #            module.bias.data.zero_()
                
    def forward(self, x):
        #x = torch.flatten(x,1) 
        return self.model(x)
    
