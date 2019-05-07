import torch
import math
#%%

class SGDOptimizer:
    def __init__(self, eta=0.1, momentum= 0.9):
        self.layer_memory = dict()
        self.eta = eta
        self.momentum = momentum 

    def __call__(self, layer):
        g = torch.cat([layer.dl_dw_cumulative, layer.dl_db_cumulative],0)

        if layer in self.layer_memory:
            u_t_1 = self.layer_memory[layer]
        else:
            u_t_1 = torch.empty(g.shape).fill_(0)
        
        w = torch.cat([layer.weight, layer.bias], 0)
        u_t = self.momentum*u_t_1 - self.eta * g
        #layer.weight = truc[:-1,:]
        #layer.bias = truc[-1,:]
        layer.weight += u_t[:-1,:]
        layer.bias += u_t[-1,:]
        self.layer_memory[layer] = u_t

class DLModule:  

    def __init__(self, *layer, optmizer = SGDOptimizer()):
        self.layer = []
        self.optmizer = optmizer
        if layer :
            for item in layer: 
                if hasattr(item, 'forward') and hasattr(item, 'backward'):
                    self.layer.append(item)  
                else: 
                    raise Exception("The specified argument should implement forward() and backward() methods")

    def __str__(self):
        value = 'Model architecture:\n'
        for layer in self.layer:
            value += str(layer) 
            if type(layer).__name__ is 'Linear':
                value += ', weights dimensions: ' + str(layer.weight.shape)
                value += ', bias dimensions: ' + str(layer.bias.shape) + '\n'
            else:
                value += '\n'
        return value

    def displayParameters(self):
        value = "Model's parameters:\n\n"
        for layer in self.layer:
            if type(layer).__name__ is 'Linear':
                value += 'Layer type: Linear ' + '\n'
                value += 'Weights: ' + str(layer.weight) + '\n'
                value += 'Bias: ' + str(layer.bias) + '\n\n' 
            else: 
                value += 'Layer type: {}'.format(type(layer).__name__) + '\n'
                value += 'None' + '\n\n'
        print(value) 
        return value 

    def sequential(self, *args):
        for item in list(args): 
            if hasattr(item, 'forward') and hasattr(item, 'backward'):
                self.layer.append(item)  
            else: 
                raise Exception("The specified argument should implement forward() and backward() methods") 

    def __call__(self, input): #def forward(self , input):
        for node in self.layer:
            #print(node)
            input = node.forward(input)
        return input
    
    def backward(self, loss:list): # NOTE - Provide a list with loss and dloss !
        output = loss[1] # dloss 
        for node in list(reversed(self.layer)):
            #print(output)
            output = node.backward(output)   
        #print(output)

    def update(self):
        for layer in self.layer:
            try:
                self.optimizer(layer) 
                #layer.zero_grad() 
            except:
                pass
    
    def zero_grad(self): # NOTE - required for training 
        for layer in self.layer:
            if type(layer).__name__ is 'Linear':
                layer.zero_grad()
'''
class Master():
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
'''

class LossMSE:

    def __init__(self):
        self.loss = None

    def __call__(self, output, target): # compute_loss - NOTE: replaced name 
        '''
            Both inputs must satisfy .view(-1,1)
            Returns a list of 2 tensors
        '''
        self.loss = self.eval(output.view(-1,1), target.view(-1,1))
        self.dloss = self.evald(output.view(-1,1), target.view(-1,1))
        return [self.loss, self.dloss]

    def eval(self, output, target):
        return torch.sum(torch.pow(output-target,2))

    def evald(self, output, target):
        return 2*(output-target)    

class Tanh:    
    def eval(self, x):
        return x.tanh()

    def evald(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

    def forward(self, input):
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        return dl_dx * self.evald(self.input)   

class Sigmoid:
    def eval(self, x):
        return torch.sigmoid(x)

    def evald(self, x):
        return torch.sigmoid(x)*(1 - torch.sigmoid(x))

    def forward(self, input):
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        return dl_dx * self.evald(self.input)  

class Relu:
    def eval(self, x):
        return x.relu()
    
    def evald(self, x):
        return (x>0).type(torch.FloatTensor)

    def forward(self, input):
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        return dl_dx * self.evald(self.input)

class Linear:
    '''
        One fully connected layer
        batch of size [n_sample x input_dim]

        after eval size [n_sample x output_dim]
        
    '''
    def __init__(self, input_dim, output_dim):
        '''
            weight size [input_dim x output_dim]
            bias size [1 x output_dim]
            grad_weight [input_dim x output_dim]
            grad_bias [1 x output_dim]
        '''

        std = math.sqrt(2.0 / (input_dim + output_dim))

        self.weight = torch.empty((input_dim, output_dim)).normal_(0, std)
        self.bias = torch.empty(1, output_dim).normal_(0, std)
        self.dl_dw_cumulative = torch.empty(input_dim, output_dim).fill_(0)
        self.dl_db_cumulative = torch.empty(1, output_dim).fill_(0)
    
    def forward(self, x):
        self.input_previous = x
        #print(x)
        return x @ self.weight + self.bias

    def backward(self, dl_ds):
        '''
            dl_dx received
            compute dl_ds = dl_dx .*dF(self.s)
            grad_weight dl_dw_cumulative += dl_ds*self.input ( input from forward pass)
            grad_bias dl_db_cumulative += dl_ds
            generate dl_dx_(l-1) for next layer : weight*dl_ds
        '''
        dl_dx = dl_ds @ self.weight.transpose(0,1)
        self.dl_dw_cumulative += self.input_previous.transpose(0,1) @ dl_ds
        self.dl_db_cumulative += dl_ds.sum(0)
        #print(self.dl_db_cumulative)
        #print(self.dl_dw_cumulative)
        return dl_dx

    def update_param(self, eta):
        self.weight -= eta*self.dl_dw_cumulative
        self.bias   -= eta*self.dl_db_cumulative 
        
    def zero_grad(self):
        self.dl_dw_cumulative.fill_(0)
        self.dl_db_cumulative.fill_(0)

class AdamOptimizer:
    def __init__(self, beta_1=0.1, beta_2=0.1, step_size=0.1, epsilon=0.1):
        self.layer_memory_m = dict()
        self.layer_memory_v = dict()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step_size = step_size
        self.epsilon = epsilon

    def step(self, layer):
        g = torch.cat([layer.dl_dw_cumulative, layer.dl_db_cumulative],0)

        if layer in self.layer_memory_g:
            m_t_1 = self.layer_memory_m[layer]
            v_t_1 = self.layer_memory_v[layer]
        else:
            m_t_1 = torch.empty(g.shape).fill_(0)
            v_t_1 = torch.empty(g.shape).fill_(0)

        m_t = self.beta_1 * m_t_1 + (1 - self.beta_1) * g
        v_t = self.beta_2 * v_t_1+ (1 - selfbeta_2) * torch.pow(g, 2)
        m_hat = m_t / (1 - self.beta_1)
        v_hat = v_t / (1 - self.beta_2)

        w = torch.cat([layer.weight, layer.bias], 0)

        truc = w - step_size * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        layer.weight = truc[:-1,:]
        layer.bias = truc[-1,:]

        self.layer_memory_m[layer] = m_t
        self.layer_memory_v[layer] = v_t
    

#%%
