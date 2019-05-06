import torch
from matplotlib import pyplot as plt 

import math
from math import pi

torch.set_grad_enabled(False)

def plotDataset(input, output):
    plt.figure()
    plt.scatter(input[output == 1][:,0],input[output == 1][:,1], label='Inside (1)')
    plt.scatter(input[output == 0][:,0],input[output == 0][:,1], label='Outside (0)') 
    plt.title('Dataset, R={}'.format(math.sqrt(1/(2*pi))))
    plt.legend()
    plt.show() 

def generate_input_output_pairs(nb):
    input = torch.rand((nb,2))*1 - 0.5
    R = math.sqrt(1/(2*pi))
    output = torch.where(input.norm(dim=1) < torch.ones(nb)*R, 
                         torch.tensor(1.),torch.tensor(0.))
     
    return input.type(torch.FloatTensor), output.type(torch.LongTensor)

def generate_dataset(nb=1000):
    train_input,train_output = generate_input_output_pairs(nb)
    test_input,test_output   = generate_input_output_pairs(nb)
    dist = train_output.sum().item()/nb*100 
    print('Distribution of training data:',dist,'%')  
    dist = test_output.sum().item()/nb*100 
    print('Distribution of testing data:',dist,'%')  
    return train_input, train_output, test_input, test_output 

# Sanity check
if __name__ == "__main__":
    train_input, train_output, test_input, test_output = generate_dataset(1000)
    plotDataset(train_input,train_output)
    plotDataset(test_input,test_output) 