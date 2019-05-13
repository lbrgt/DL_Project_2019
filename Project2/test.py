#%%
import sys
import os

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
    Linear(25,25),
    Relu(),
    Linear(25,2),
    Sigmoid(),
    optimizer=SGDOptimizer(eta=0.001, momentum= 0.5)
)
# Display its architecture
print(model)
# Display its parmeters
model.displayParameters()
#%%

# Visualize the model's initial behaviour 
showBehaviour = lambda model,input: dg.plotDataset(input,model(input).argmax(1).view(-1)) 
#showBehaviour(model,test_input)

#%%
# Define a loss - NOTE: only has to be instantiated once now
criterion = LossMSE()

train_target
train_target_OH = torch.empty((train_target.shape[0],int(train_target.max().item()+1))).fill_(0.)
train_target_OH[range(train_target.shape[0]), train_target.type(torch.LongTensor)] = 1

# %%
# Train the network
epochs = 100
mini_batch_size = 200
for e in range(epochs):
    sum_loss = 0
    # We do this with mini-batches
    for b in range(0, train_input.size(0), mini_batch_size):

        output = model(train_input.narrow(0, b, mini_batch_size))
        loss = criterion(output, train_target_OH.narrow(0, b, mini_batch_size))
        sum_loss = sum_loss + loss[0].item() # NOTE - loss a bit dirty but okay? 
        model.zero_grad()
        model.backward(loss)
        model.update() 
         
    print(e, '-', sum_loss)

print(model(test_input).argmax(1).view(-1, 1))
showBehaviour(model,test_input)    

print(torch.sum(model(test_input).argmax(1).int()==test_target.int()).item())
#%%
