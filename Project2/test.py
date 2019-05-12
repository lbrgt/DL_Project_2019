#%%
import sys
import os

sys.path.append(os.getcwd()+'/Project2/')

import torch
from NeuralNet import DLModule, LossMSE, Tanh, Sigmoid, Relu, Linear, SGDOptimizer, AdamOptimizer
import data_generator as dg 

# Disable torch's gradient functionnalities 
torch.set_grad_enabled(False)

# Generate the dataset 
num_samples = 1000
train_input, train_target, test_input, test_target = dg.generate_dataset(num_samples) 

# Instanciate a model 
model = DLModule(
    Linear(2,3),
    Relu(),
    Linear(3,1),
    Sigmoid(),
    optimizer=SGDOptimizer(eta=0.001,momentum=0.00)
)
# Display its architecture
print(model)
# Display its parmeters
model.displayParameters()
#%%

# Visualize the model's initial behaviour 
showBehaviour = lambda model,input: dg.plotDataset(input,(model(input) < 0.5).int().view(-1)) 
#showBehaviour(model,test_input)

#%%

# Define a loss - NOTE: only has to be instantiated once now
criterion = LossMSE()

# Train the network
epochs = 500
mini_batch_size = 100
for e in range(epochs):
    sum_loss = 0
    # We do this with mini-batches
    for b in range(0, train_input.size(0), mini_batch_size):
        output = model(train_input.narrow(0, b, mini_batch_size))
        loss = criterion(output, train_target.narrow(0, b, mini_batch_size).view(-1,1))
        sum_loss = sum_loss + loss[0].item() # NOTE - loss a bit dirty but okay? 
        model.zero_grad()
        model.backward(loss)  
        model.update() 
         
    print(e, '-', sum_loss)


showBehaviour(model,test_input)    

#%%

# Instanciate a model 
model_test = DLModule(
    Linear(2,1),
    Relu(),
    optimizer=SGDOptimizer(eta=0.1,momentum=0.0)
)


model_test.displayParameters()

out = model_test(torch.Tensor([[1,1],[2,2]]))
criterion = LossMSE()

loss  = criterion(out, torch.Tensor([[1.0000],[1.0000]]))

model_test.backward(loss)
print(model_test.layer[0].weight)



model_test.update()


#%%

print(model_test.layer[2].dl_dw_cumulative)


#%%
out = model_test(torch.Tensor([[1,1],[2,2]]))

loss  = criterion(out, torch.Tensor([[1.0000],[1.0000]]))

model_test.backward(loss)
print(model_test.layer[0].weight)



model_test.update()

#%%
print(model_test.layer[0].weight)


#%%


#%%


#%%
