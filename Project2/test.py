#%%
import sys
import os
sys.path.append(os.getcwd()+'/Project2/')

import torch
from NeuralNet import DLModule, LossMSE, Tanh, Sigmoid, Relu, Linear, SGDOptimizer, AdamOptimizer
import data_generator as dg 

# Disable torch functionnalities 
torch.set_grad_enabled(False)

# Generate the dataset 
num_samples = 1000
train_input, train_target, test_input, test_target = dg.generate_dataset(num_samples) 

# Instanciate the model
eta = 0.1 
momentum = 0.9
model = DLModule(
    Linear(2,3),
    Tanh(),
    Linear(3,1),
    Sigmoid(),
    optmizer=SGDOptimizer(eta,momentum)
)
# Display its architecture
print(model)
# Display its parmeters
model.displayParameters()

# Visualize model's initial behaviour with this lambda function
showBehaviour = lambda model,input: dg.plotDataset(input,(model(test_input) < 0.5).int().view(-1)) 
#showBehaviour(model,test_input)

# Define a loss - NOTE: only has to be instantiated once now
criterion = LossMSE()

# Train the network
epochs = 5
mini_batch_size = 100
for e in range(epochs):
    sum_loss = 0
    # We do this with mini-batches
    for b in range(0, train_input.size(0), mini_batch_size):
        output = model(train_input.narrow(0, b, mini_batch_size))
        loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
        # print(loss[0].item(), loss[1])
        sum_loss = sum_loss + loss[0].item()
        model.zero_grad()
        model.backward(loss)  
        model.update() 
    #print(e, '-', sum_loss)


#showBehaviour(model,test_input)    




'''
# Define a batch of training samples and do a forward pass 
train_input = torch.Tensor([[1,1],[2,2]])
train_output = model.forward(train_input)
print(train_output)

# Define a batch of training targets
train_target = torch.Tensor([[1,1,1],[2,2,2]])



# Compute the loss 
loss = lossMSE.compute_loss(train_target, train_output) 

# Backward pass
model.backward(loss) # NOTE - Should only provide values, not instance! 

# Update the model
model.update(eta=0.1)
'''
#%%
'''
b = torch.Tensor([[1,1],[2,2]])
output = nn.forward_pass(b)

#%%
target = torch.Tensor([[1,1,1],[2,2,2 ]])

lossMSE = LossMSE()

lossMSE.compute_loss(target, output)

nn.backward_pass(lossMSE)
#%%
nn.update(0.1)


#%%
nn.layer[0].zero_grad()
nn.layer[0].dl_db_cumulative
#%%
'''