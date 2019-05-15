#%%
import sys
import os
import time

sys.path.append(os.getcwd()+'/Project2/')

import torch
from NeuralNet import DLModule, CrossEntropyLoss, LossMSE, Tanh, Sigmoid, Relu, Linear, SGDOptimizer, AdamOptimizer
import data_generator as dg 

# Disable torch's gradient functionnalities 
torch.set_grad_enabled(False)

# Generate the dataset 
num_samples = 1000
train_input, train_target, test_input, test_target = dg.generate_dataset(num_samples) 

# Instanciate a model 
model = DLModule(
    Linear(2,25),
    Relu(),
    Linear(25,25),
    Relu(),
    Linear(25,2),
    Sigmoid(),
    optimizer= SGDOptimizer(learning_rate=0.01, momentum= 0.5, decay=0.01) # AdamOptimizer( beta_1=0.9, beta_2=0.99, step_size=0.001, epsilon=0.0000001) 
)

# Display its architecture
#print(model)

# Display its parmeters
model.displayParameters()
#%%

# Visualize the model's initial behaviour 
showBehaviour = lambda model,input: dg.plotDataset(input,model(input).argmax(1).view(-1)) 

#%%
# Define a loss -Has to be instantiated only once
criterion = LossMSE() # OPTION: CrossEntropyLoss()

# Convert the target generated with label into a one hot encoded vector for the MSE Loss
train_target
train_target_OH = torch.empty((train_target.shape[0],int(train_target.max().item()+1))).fill_(0.)
train_target_OH[range(train_target.shape[0]), train_target.type(torch.LongTensor)] = 1

#input("Press enter to continue")

# Train the network
# The syntax is as close as possible as pytorch
epochs = 100
mini_batch_size = 100

start = time.time()
for e in range(epochs):
    sum_loss = 0
    # We do this with mini-batches
    for b in range(0, train_input.size(0), mini_batch_size):

        output = model(train_input.narrow(0, b, mini_batch_size))
        loss = criterion(output, train_target_OH.narrow(0, b, mini_batch_size))
        sum_loss = sum_loss + loss[0].item()
        model.zero_grad()
        model.backward(loss)
        model.update(e) 
         
    print(e, '-', sum_loss)

# Compute the learning time
stop = time.time()
duration = stop-start
print("This model, with {} epoch and a batch size of {}, was trained in {:.2f}s".format(epochs, mini_batch_size, duration))

# Compute the error for the train set
train_output = model(train_input)
nbr_error = torch.sum(train_output.argmax(1).int()!=train_target.int()).item()
print('After training, the ratio of missclassified point for the train set is {} % '.format(100*nbr_error/test_target.shape[0]))

# Compute the error for the test set
test_output = model(test_input)
nbr_error = torch.sum(test_output.argmax(1).int()!=test_target.int()).item()
print('After training, the ratio of missclassified point for the test set is {} % '.format(100*nbr_error/test_target.shape[0]))

# Plot the output of the model for the test set
showBehaviour(model,test_input)    