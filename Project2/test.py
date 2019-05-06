#%%
import sys
import os
sys.path.append(os.getcwd()+'/Project2/')

import torch
from NeuralNet import DLModule, LossMSE, Tanh, Relu, Linear, SGDOptimizer, AdamOptimizer

model = DLModule(optmizer=SGDOptimizer())
model.sequential(Linear(2,3), Relu(), Linear(3,3), Relu())
print(model)
#model.optmizer = AdamOptimizer(0.1, 0.1, 0.1, 0.1)



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