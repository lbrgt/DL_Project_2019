# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# http://www.graphviz.org/download/
# To use the computation graph dot.exe mlp.dot -Lg -T pdf -o mlp.pdf
# where mlp.dot was generated using agtree2dot.save_dot(loss,{},open('./mlp.dot', 'w'))
# and mlp.pdf is the pdf file to create
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

# Import the dataset
import dlc_practical_prologue as prologue
nb = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

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
        x = F.relu(self.fc3(x)) 
        return x 




class BasicNet(nn.Module):
    '''
        Net caracteristics:
            - No weight sharing
            - No intermediate loss
        
        Net input: Nx2x14x14
        Net output: 2x1 
    '''
    def __init__(self):
        super(BasicNet, self).__init__()
        self.parallel_net1 = Parallel_Net()
        self.parallel_net2 = Parallel_Net()
        self.analyser_net  = Analyzer_Net()
    
    def forward(self,x):
        # Split the 2 input channels
        x1 = x[:,0,:,:].view(-1,1,14,14)
        x2 = x[:,1,:,:].view(-1,1,14,14)

        # No weight sharing (declare 2 distinct instances of Parallel_Net)
        x1 = self.parallel_net1(x1)
        x2 = self.parallel_net2(x2)

        # Concatenate back both classification results 
        x = torch.cat((x1.view(-1,10),x2.view(-1,10)),dim=1)
        x = self.analyser_net(x)

        return x 

# Define the mini_batch size
batch_size = 1

# Create an instance of the network
basicModel = BasicNet()

# Create losses
mseLoss = nn.MSELoss()
crossEnt = nn.CrossEntropyLoss()

# Run a forward pass and observe the computational graph
output = basicModel(train_input.narrow(0,0,batch_size))
train_target = train_target.type(torch.FloatTensor)

loss = mseLoss(output, train_target.narrow(0, batch_size, batch_size))
basicModel.zero_grad()
loss.backward()
# Save the computation tree for later rendering using dot.exe
ag.save_dot(loss,{},open('./mlp.dot', 'w')) 

