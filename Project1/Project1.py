#%%

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

nb = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

print(train_input.size())

#%%
print(1)

