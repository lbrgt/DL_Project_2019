import torch
from torch import Tensor, optim
import dlc_practical_prologue as prologue
from matplotlib import pyplot as plt
import time

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from sub_modules import Parallel_Net, Analyzer_Net, evaluateClassIdentification, generateComputationalTree

####################################################################################################
'''
Construction of the basic net with :

    - Weight sharing

    - Intermediate/auxiliary loss

Definition of the corresponding training function, test of the resulting net.    

'''
####################################################################################################

# Load the dataset
nb = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

        return x , x1, x2


def train_network(model, train_input, train_target, train_classes, mini_batch_size,w=0.5):
    train_target_One_Hot = torch.eye(2)[train_target]

    # Specify the loss function
    criterion_total = nn.MSELoss()
    criterion_classifier = nn.CrossEntropyLoss()

    # Define the number of epochs to train the network
    epochs = 25

    # Set the learning rate
    eta = 0.01

    #Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    loss_record_total=[]
    loss_record_classifier=[]
    for e in range(epochs):
        sum_loss_total = 0
        sum_loss_classifier = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            model.zero_grad()
            output, x1, x2 = model(train_input.narrow(0, b, mini_batch_size)) # dim,start,length
            y = [x1, x2]
            l =[0, 1]
            for x,i in zip(y,l):
                # For each image (since there are 2 channels)
                loss_classifier = w*criterion_classifier(x , train_classes[:,i].narrow(0, b, mini_batch_size))
                sum_loss_classifier += loss_classifier.item()
                loss_classifier.backward(retain_graph=True)

            loss_total = (1-w)*criterion_total(output, train_target_One_Hot.narrow(0, b, mini_batch_size))
            sum_loss_total += loss_total.item()
            loss_total.backward()

            optimizer.step()
        loss_record_total.append(sum_loss_total)
        loss_record_classifier.append(sum_loss_classifier)
        #print('Sum of classifier loss at epoch {}: \t'.format(e),sum_loss_classifier)  
        #print('Sum of total loss at epoch {}: \t'.format(e),sum_loss_total)  

    return model, loss_record_total , loss_record_classifier


      
def evaluateFinalOutput(model, test_input, test_target, mini_batch_size):
    test_target = test_target.type(torch.FloatTensor)
    
    with torch.no_grad():
        error = 0
        for b in range(0, test_input.size(0), mini_batch_size):
            output, _, _ = model(test_input.narrow(0, b, mini_batch_size))
            for i in range(output.size(0)):
                if torch.argmax(output[i]) == 1:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() < 0.2:
                        error += 1
                elif torch.argmax(output[i]) == 0:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() > 0.8:
                        error += 1
                else:
                    error += 1
    return error/test_target.size(0)*100    

def main():
    # Define the mini_batch size (A PLACER DANS LE MASTER)
    mini_batch_size = 100
    # Run grid search on weighting on the losses - 0.5 seems best
    for w in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        # Create an instance of the network
        basicModel = Net()
        num_param = sum(p.numel() for p in basicModel.parameters() if p.requires_grad)
        #print('Number of trainable parameters:',num_param)  

        # Train the network
        basicModel, _, _ = train_network(basicModel,train_input, train_target, train_classes, mini_batch_size,w=w)
        #print(type(basicModel))

        # Evaluate the performance of the model
        res = evaluateFinalOutput(basicModel,test_input,test_target,mini_batch_size)
        print('Error rate of the model: ',res,'%, weight:', w) 

if __name__ == "__main__":
    main()



