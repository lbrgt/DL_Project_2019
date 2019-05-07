import torch
from matplotlib import pyplot as plt 
import numpy as np 
from numpy import pi 

torch.set_grad_enabled(False)

def plotDataset(input, output):
    plt.figure()

    # Draw desired boundary
    R = np.sqrt(1/(2*pi))
    t = np.linspace(0, 2*pi, 100)
    x = R*np.cos(t)
    y = R*np.sin(t)
    plt.plot(x,y,color='g',linestyle='--') 

    # Draw generated data points 
    plt.scatter(input[output == 1][:,0],input[output == 1][:,1], label='Inside (1)')
    plt.scatter(input[output == 0][:,0],input[output == 0][:,1], label='Outside (0)') 
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Dataset, R={}'.format(R))
    plt.legend()
    plt.tight_layout()
    plt.grid()  
    plt.show() 

def generate_input_output_pairs(nb):
    input = torch.rand((nb,2))*1 - 0.5
    R = np.sqrt(1/(2*pi))
    output = torch.where(input.norm(dim=1) < torch.ones(nb)*R, 
                         torch.tensor(1.),torch.tensor(0.))
     
    return input.type(torch.FloatTensor), output.type(torch.FloatTensor)

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
    print('Input and output sizes:',train_input.shape, train_output.shape)
    plotDataset(train_input,train_output)
    plotDataset(test_input,test_output) 
