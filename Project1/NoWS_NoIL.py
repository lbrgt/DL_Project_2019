import torch
from torch import Tensor
import dlc_practical_prologue as prologue
from matplotlib import pyplot as plt
import time

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from sub_modules import Parallel_Net, Analyzer_Net, \
        evaluateClassIdentification, evaluateFinalOutput, generateComputationalTree

####################################################################################################
'''
Construction of the basic net with :

    - No weight sharing

    - No intermediate/auxiliary loss

Definition of the corresponding training function, test of the resulting net.    

'''
####################################################################################################

# Load the dataset
nb = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


def train_network(model, train_input, train_target, mini_batch_size):
    train_target_One_Hot = torch.eye(2)[train_target]
    # Specify the loss function
    criterion = nn.MSELoss()

    # Define the number of epochs to train the network
    epochs = 25
    
    # Set the learning rate
    eta = 0.1
    loss_record=[]
    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size)) # dim,start,length
            loss = criterion(output, train_target_One_Hot.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        loss_record.append(sum_loss)
        print('Sum of loss at epoch {}: \t'.format(e),sum_loss)
    return model, loss_record


# Define the mini_batch size (A PLACER DANS LE MASTER)
mini_batch_size = 100

# Create an instance of the network
basicModel = Net()
num_param = sum(p.numel() for p in basicModel.parameters() if p.requires_grad)
print('Number of trainable parameters:',num_param)  

# Train the network
basicModel = train_network(basicModel,train_input, train_target, mini_batch_size)


# Evaluate the performance of the model
res = evaluateFinalOutput(basicModel,test_input,test_target,mini_batch_size)
print('Error rate of the model: ',res,'%')





