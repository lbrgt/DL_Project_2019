import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

nb = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

class Basic_Net(nn.Module):
    def __init__(self):
        super(Basic_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(10*14*14, 10)
        self.fc2 = nn.Linear(20,2)


    def forward(self, x):
       
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x0 = F.relu(self.conv1(x0))
        x0 = self.fc1(x0.view(-1, 10*14*14))

        x1 = x[:,1,:,:].view(-1,1,14,14)
        x1 = F.relu(self.conv1(x1))
        x1 = self.fc1(x1.view(-1, 10*14*14))
        x_f = torch.empty(20)
        x_f[0:10]=x0
        x_f[10:20]=x1
        x_f=self.fc2(x_f)
        #x_f = (torch.argmax(x0) - torch.argmax(x1)).sign()

        return x_f

batch_size = 10
model = Basic_Net() 
x_f = model(train_input.narrow(0, 0, batch_size))

print(x_f.item())


