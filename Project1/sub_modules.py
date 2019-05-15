# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import sys
import os
#sys.path.append(os.getcwd()+'/Project1/')


import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

# Define a Net that works with a single image (hence channel dim = 1)
class Parallel_Net(nn.Module):
    def __init__(self):
        super(Parallel_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*3*3, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(-1, 64*3*3)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

class Analyzer_Net(nn.Module):
    def __init__(self):
        super(Analyzer_Net, self).__init__()
        self.fc1 = nn.Linear(2*10, 50)   
        self.fc2 = nn.Linear(50, 35)
        self.fc3 = nn.Linear(35, 10)
        self.fc4 = nn.Linear(10, 6)
        self.fc5 = nn.Linear(6, 2) 
        
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

