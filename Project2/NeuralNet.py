import torch
import math
#%%

class SGDOptimizer:
    '''
        Stochastic Gradient Descent Optimizer update the weight and bias of the corresponding layers.
        The learning rate, momentum and decay rate can be defined. 
    ''' 
    def __init__(self, learning_rate=0.01, momentum= 0.9, decay = 0.0):

        self.layer_memory = dict() # Store previous cumulative gradient
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay

    def __call__(self, layer, epoch):
        # Concatenate to ease the computation
        g = torch.cat([layer.dl_dw_cumulative, layer.dl_db_cumulative],0)

        # Retrieve previous cumulative gradient
        if layer in self.layer_memory:
            u_t_1 = self.layer_memory[layer]
        else:
            u_t_1 = torch.empty(g.shape).fill_(0)

        # Decreasing learning rate to allow fast and non-oscillating convergence
        lrate = self.learning_rate * (1 / (1 + self.decay * epoch))

        # Momentum implementation     
        u_t = self.momentum*u_t_1 + lrate * g

        layer.weight = layer.weight - u_t[:-1,:] 
        layer.bias = layer.bias - u_t[-1,:]     
        self.layer_memory[layer] = u_t

class AdamOptimizer:
    '''
        Adam Optimizer update the weight and bias of the corresponding layers.
    ''' 
    def __init__(self, beta_1=0.9, beta_2=0.99, step_size=0.001, epsilon=0.000001):
        self.layer_memory_m = dict()    # Store values from previous step
        self.layer_memory_v = dict()    # Store values from previous step
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step_size = step_size
        self.epsilon = epsilon

    def __call__(self, layer):

        # Concatenate to ease the computation       
        g = torch.cat([layer.dl_dw_cumulative, layer.dl_db_cumulative],0)

        # Retrives values from previous step
        if layer in self.layer_memory_v:
            m_t_1 = self.layer_memory_m[layer]
            v_t_1 = self.layer_memory_v[layer]
        else:
            m_t_1 = torch.empty(g.shape).fill_(0)
            v_t_1 = torch.empty(g.shape).fill_(0)

        # Applies Adam formula
        m_t = self.beta_1 * m_t_1 + (1 - self.beta_1) * g
        v_t = self.beta_2 * v_t_1+ (1 - self.beta_2) * torch.pow(g, 2)
        m_hat = m_t / (1 - self.beta_1)
        v_hat = v_t / (1 - self.beta_2)
        w = torch.cat([layer.weight, layer.bias], 0)
        A = w - self.step_size * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        layer.weight = A[:-1,:]
        layer.bias = A[-1,:].view(1,-1)
        self.layer_memory_m[layer] = m_t
        self.layer_memory_v[layer] = v_t
        
class DLModule: 
    '''
        Main module which embeds the structure of the neural network and provide function to deal with it. Can proccess batch of sample.
    '''  

    def __init__(self, *layer, optimizer = SGDOptimizer()):
        self.layer = []
        self.optimizer = optimizer
        if layer :
            self.sequential(layer)

    # Print overload to easily display the stored architecture
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
        '''
            Display the network's parameters values
        '''
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

    def sequential(self, layer):
        '''
            Create the structure of the network by storing the different layers. They have to implement bacward and forward functions.
        '''
        for item in layer: 
            if hasattr(item, 'forward') and hasattr(item, 'backward'):
                self.layer.append(item)  
            else: 
                raise Exception("The specified argument should implement forward() and backward() methods") 

    def __call__(self, input):
        '''
            Following Pytorch notation, compute one forward pass trough the network.
        '''  
        for node in self.layer:
            input = node.forward(input)
        return input
    
    def backward(self, loss:list): 
        '''
            Compute one backward pass trough the network, updating the cumulative gradient of the different layers. 
        '''
        output = loss[1]  
        for node in list(reversed(self.layer)):
            output = node.backward(output)   

    def update(self, epoch):
        '''
            Update the parameters of the different layers, usinig the Optimizer defined. It use the cumulative gradient value stored on the different layer. 
        '''
        for layer in self.layer:
            try:
                self.optimizer(layer, epoch) 
            except Exception as e: 
                pass
    
    def zero_grad(self):
        '''
            Reset the cumulative gradient values to zero.
        '''
        for layer in self.layer:
            if type(layer).__name__ is 'Linear':
                layer.zero_grad()

class LossMSE:
    '''
        Implement the Mean Squared Error Loss. Compute loss and the derivative of it to backpropagate.  
    '''
    def __init__(self):
        self.loss = None

    def __call__(self, output, target): 
        '''
            Both output and target must satisfy .view(-1,"size of sample").
            If classification, the "target" tensor has to be one hot encoded.
            Returns a list of 2 tensors [loss dloss]. 
        '''
        self.loss = self.eval(output, target)
        self.dloss = self.evald(output, target)
        return [self.loss, self.dloss]

    def eval(self, output, target):
        return torch.sum(torch.pow(output-target,2))

    def evald(self, output, target):
        return 2*(output-target)    

class Tanh: 
    '''
        Implement tanh activation function. Implement forward() and backward().
    '''   
    def eval(self, x):
        return x.tanh()

    def evald(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

    def forward(self, input):
        '''
            Store the input tensor to be used during the backward pass. 
        '''
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        return dl_dx * self.evald(self.input)   

class Sigmoid: 
    '''
        Implement sigmoid activation function. Implement forward() and backward().
    '''       
    def eval(self, x):
        return torch.sigmoid(x)

    def evald(self, x):
        return torch.sigmoid(x)*(1 - torch.sigmoid(x))

    def forward(self, input):
        '''
            Store the input tensor to be used during the backward pass. 
        '''
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        return dl_dx * self.evald(self.input)  

class Relu:
    '''
        Implement Relu activation function. Implement forward() and backward().
    '''  
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
        
        # Better initialisation to reduce the vanishing of the gradient
        std = math.sqrt(2.0 / (input_dim + output_dim))
        self.weight =           torch.empty((input_dim, output_dim)).normal_(0, std)
        self.bias =             torch.empty(1, output_dim).normal_(0, std)

        self.dl_dw_cumulative = torch.empty((input_dim, output_dim)).fill_(0)
        self.dl_db_cumulative = torch.empty(1, output_dim).fill_(0)
        

    def forward(self, x):
        self.input_previous = x
        return x @ self.weight + self.bias

    def backward(self, dl_ds):
        dl_dx = dl_ds @ self.weight.transpose(0,1)
        self.dl_dw_cumulative += self.input_previous.transpose(0,1) @ dl_ds # Sums up over the whole batch
        self.dl_db_cumulative += dl_ds.sum(0)                               # Sums up over the whole batch
        return dl_dx

    def update_param(self, eta):
        self.weight -= eta*self.dl_dw_cumulative
        self.bias   -= eta*self.dl_db_cumulative 
        
    def zero_grad(self):
        self.dl_dw_cumulative.fill_(0)
        self.dl_db_cumulative.fill_(0)

class CrossEntropyLoss():
    '''
        Implement the Cross entropy Loss. 
        Compute loss and the derivative of it to backpropagate.  
    '''
    def __init__(self):
        self.loss = None
        self.dloss = None

    def __call__(self, output, target): 
        '''
            Both output and target must satisfy .view(-1,"size of sample")
            The "target" tensor has to contain the label [0, nbr_class-1].
            Returns a list of 2 tensors [loss dloss] 
        '''
        self.loss = self.eval(output, target)
        self.dloss = self.evald(output, target)
        return [self.loss, self.dloss]

    def softmax(self, T):
        exps = torch.exp(T - torch.max(T, 1)[0].view(-1,1)) # Avoid log extrema
        return exps/torch.sum(exps,1).view(-1,1)
    
    def eval(self, output, target):
        proba = self.softmax(output)
        n_sample = target.shape[0]
        log_likelihood = -torch.log(proba[range(n_sample), target.type(torch.LongTensor).view(1,-1)])
        return torch.sum(log_likelihood) / n_sample

    def evald(self, output, target):
        proba = self.softmax(output)
        n_sample = target.shape[0]
        proba[range(n_sample),target.type(torch.LongTensor)] -= 1
        return proba

