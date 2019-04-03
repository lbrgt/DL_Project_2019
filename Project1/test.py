#%%
%load dlc_practical_prologue

#%%
import torch

import sys
import os
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
sys.path.append(os.getcwd()+'DL_Project_2019/Project1/')
   #  '/home/lburget/Documents/EPFL/Master/DeepLearning/Project/DL_Project_2019/Project1/')
import dlc_practical_prologue as prologue

print()
nb = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

print(train_input.size())
#%%
print(1)

