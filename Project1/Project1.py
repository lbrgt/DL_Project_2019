# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

nb = 1000

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

class Parallel_Net(nn.Module):
    # Input is Nx2x14x14
    def __init__(self):
        super(Parallel_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(10*14*14, 10)

    def sharedPass(self,x):
        x = F.relu(self.conv1(x))
        x = self.fc1(x.view(-1, 10*14*14))
        return x 

    def forward(self, x):
        # Split the 2 input channels
        x0 = x[:,0,:,:].view(-1,1,14,14)
        x1 = x[:,1,:,:].view(-1,1,14,14)

        # Run the shared weights section 
        x0 = self.sharedPass(x0)
        x1 = self.sharedPass(x1)

        return x0,x1

class Analyzer_Net(nn.Module):
    # Input is Nx20
    def __init__(self):
        super(Analyzer_Net, self).__init__()
        self.fc1 = nn.Linear(2*10, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2) 
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        return x 

batch_size = 100
model_parallel = Parallel_Net() 
model_analyzer = Analyzer_Net()
mseLoss = nn.MSELoss()
crossEnt = nn.CrossEntropyLoss()

x0,x1 = model_parallel(train_input.narrow(0, 0, batch_size))
x = torch.cat((x0.view(-1),x1.view(-1)))
result = model_analyzer(x)








class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
def runExample():
    train_input, train_target, test_input, test_target = \
        prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)
    train_input, train_target = Variable(train_input), Variable(train_target)

    model, criterion = Net(), nn.MSELoss()
    eta, mini_batch_size = 1e-1, 100

    for e in range(0, 25):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)