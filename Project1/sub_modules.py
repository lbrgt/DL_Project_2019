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


# Evaluate the network's performance with winner takes it all approach
def evaluateClassIdentification(model, test_input, test_classes, mini_batch_size):
    error = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input[:,0].view(-1,1,14,14).narrow(0, b, mini_batch_size))
        
        c_array = output.argmax(1)
        t_array = test_classes[:,0][b:b+mini_batch_size]
        error += (c_array-t_array).nonzero().size()[0]
        
    return error/test_input.size()[0]*100

#Generation of the computational tree
def generateComputationalTree():
    global train_input
    basicModel = BasicNet()
    output = basicModel(train_input.narrow(0, 0, 1))
    criterion = nn.MSELoss()
    loss = criterion(output, train_target.narrow(0, 0, 1))
    loss.backward()
    # Save the computation tree for later rendering using dot.exe
    ag.save_dot(loss,{},open('./mlp.dot', 'w')) 

