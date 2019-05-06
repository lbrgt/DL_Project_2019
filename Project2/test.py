#%%
import sys
import os
sys.path.append(os.getcwd()+'/Project2/')

import torch
from NeuralNet import DLModule, LossMSE, Tanh, Relu, Linear, SGDOptimizer, AdamOptimizer

# Instanciate the model
model = DLModule(
    Linear(2,3),
    optmizer=SGDOptimizer(),
)
# Define its architecture
#model.sequential(Linear(2,3), Relu(), Linear(3,3), Relu())
# Display its architecture
print(model)

# Define a batch of training samples and do a forward pass
train_input = torch.Tensor([[1,1],[2,2]])
train_output = model.forward(train_input)

# Define a batch of training targets
train_target = torch.Tensor([[1,1,1],[2,2,2]])

# Define a loss 
lossMSE = LossMSE()

# Compute the loss 
loss = lossMSE.compute_loss(train_target, train_output) 

# Backward pass
model.backward(lossMSE) # NOTE - Should only provide values, not instance! 

# Update the model
model.update(eta=0.1)

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