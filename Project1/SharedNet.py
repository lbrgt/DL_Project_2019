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
# Load the dataset
nb = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

# Import the building blocks of the network
from sub_modules import Parallel_Net,Analyzer_Net


class BasicNet(nn.Module):
    '''
        Net caracteristics:
            - Weight sharing
            - No intermediate loss
        
        Net input: Nx2x14x14
        Net output: 2x1 
    '''
    def __init__(self):
        super(BasicNet, self).__init__()
        self.parallel_net = Parallel_Net()
        self.analyser_net  = Analyzer_Net()
    
    def forward(self,x):
        # Split the 2 input channels
        x1 = x[:,0,:,:].view(-1,1,14,14)
        x2 = x[:,1,:,:].view(-1,1,14,14)

        # No weight sharing (declare 2 distinct instances of Parallel_Net)
        x1 = self.parallel_net(x1)
        x2 = self.parallel_net(x2)

        # Concatenate back both classification results 
        x = torch.cat((x1.view(-1,10),x2.view(-1,10)),dim=1)
        x = self.analyser_net(x)

        return x 

def train():
    global train_input, train_target, train_classes
    train_target = train_target.type(torch.FloatTensor)

    # Define the mini_batch size
    batch_size = 200

    # Create an instance of the network
    basicModel = BasicNet()
    num_param = sum(p.numel() for p in basicModel.parameters() if p.requires_grad)
    print('Number of trainable parameters:',num_param)

    # Specify the loss function
    criterion = nn.MSELoss()

    # Define the number of epochs to train the network
    epochs = 100
    # Set the learning rate
    eta = 0.1

    for e in range(0, epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            #print('input: ',train_input.narrow(0, b, batch_size))
            output = basicModel(train_input.narrow(0, b, batch_size))
            #print('output: ', output)
            loss = criterion(output, train_target.narrow(0, b, batch_size))
            sum_loss = sum_loss + loss.item()
            basicModel.zero_grad()
            """ basicModel.parallel_net1.zero_grad()
            basicModel.parallel_net2.zero_grad()
            basicModel.analyser_net.zero_grad() """
            loss.backward()
            for p in basicModel.parameters():
                p.data.sub_(eta * p.grad.data)
            """ for p in basicModel.parallel_net1.parameters():
                p.data.sub_(eta * p.grad.data)
            for p in basicModel.parallel_net2.parameters():
                p.data.sub_(eta * p.grad.data)
            for p in basicModel.analyser_net.parameters():
                p.data.sub_(eta * p.grad.data) """
        #return
        print('Sum of loss at epoch {}: \t'.format(e),sum_loss)
    
    res = evaluate(basicModel,batch_size=batch_size)
    print('Error rate of BasicNet: ',res,'%')


def evaluate(model,batch_size=100):
    global test_input, test_target, test_classes
    test_target = test_target.type(torch.FloatTensor)

    with torch.no_grad():
        error = 0
        for b in range(0, test_input.size(0), batch_size):
            output = model(test_input.narrow(0, b, batch_size))
            for i in range(output.size(0)):
                #print(output[i].item(),test_target.narrow(0, b, batch_size)[i].item())
                if output[i].item() >= 0.5:
                    if test_target.narrow(0, b, batch_size)[i].item() < 0.2:
                        error += 1
                elif output[i].item() < 0.5:
                    if test_target.narrow(0, b, batch_size)[i].item() > 0.8:
                        error += 1
                else:
                    error += 1
    return error/test_target.size(0)*100


def generateComputationalTree():
    global train_input
    basicModel = BasicNet()
    output = basicModel(train_input.narrow(0, 0, 1))
    criterion = nn.MSELoss()
    loss = criterion(output, train_target.narrow(0, 0, 1))
    loss.backward()
    # Save the computation tree for later rendering using dot.exe
    ag.save_dot(loss,{},open('./mlp.dot', 'w')) 


if __name__ == "__main__":
    train()
    
