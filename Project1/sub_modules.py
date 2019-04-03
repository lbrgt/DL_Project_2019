# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import agtree2dot as ag 

#%%
import sys
import os
sys.path.append(os.getcwd()+'/Project1/')
#%%

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

# Define a Net that works with a single image (hence channel dim = 1)
class Parallel_Net(nn.Module):
    # Input is Nx1x14x14
    '''
        Building block used to classify the digit.

        Net input: Nx1x14x14 (single image)
        Net output: 10x1
    '''
    def __init__(self):
        super(Parallel_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(10*14*14, 10)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.fc1(x.view(-1, 10*14*14))
        return x 

class Analyzer_Net(nn.Module):
    '''
        Building block used to infer digit's relation.

        Net input: 20x1 (digit classification)
        Net output: 1x1 (bigger or smaller)
    '''
    def __init__(self):
        super(Analyzer_Net, self).__init__()
        self.fc1 = nn.Linear(2*10, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1) 
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x)) 
        return x 
