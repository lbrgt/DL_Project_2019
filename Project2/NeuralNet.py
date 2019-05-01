import torch
#%%
class DLModule:  

    def __init__(self, *args):
        self.layer = []
        print(args)
        if args :
            for item in list(args): 
                if hasattr(item, 'forward') and hasattr(item, 'backward'):
                    self.layer.append(item)  
                else: 
                    raise Exception("The module should containt forward and backward pass")

    def sequential(self, *args):
        for item in list(args): 
            print('ici')
            print(item)
            if hasattr(item, 'forward') and hasattr(item, 'backward'):
                self.layer.append(item)  
            else: 
                raise Exception("The module should containt forward and backward pass")
        print(self.layer)

    def forward_pass(self , input):
        for node in self.layer:
            print(node)
            input = node.forward(input)
        return input
    
    def backward_pass(self, loss):
        output = loss.dloss
        for node in list(reversed(self.layer)):
            print(output)
            output = node.backward(output)   
        print(output)

    def update(self,eta): 
        for node in self.layer:
            try:
                node.update_param(eta)
                node.zero_grad()
            except:
                pass
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

    def compute_loss(self, output, target):
        self.loss = self.eval(output, target)
        self.dloss = self.evald(output, target)
        return self.loss

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
            bias size [ 1 x output_dim]
            grad_weight [ input_dim x output_dim]
            grad_bias [ 1 x output_dim]
        '''

        std = math.sqrt(2.0 / (input_dim + output_dim))

        self.weight = torch.empty((input_dim, output_dim)).normal_(0, std)
        self.bias = torch.empty(1, output_dim).normal_(0, std)
        self.dl_dw_cumulative = torch.empty(input_dim, output_dim).fill_(0)
        self.dl_db_cumulative = torch.empty(1, output_dim).fill_(0)
    
    def forward(self, x):
        self.input_previous = x
        print('La')
        print(x)
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
        print(self.dl_db_cumulative)
        print(self.dl_dw_cumulative)
        return dl_dx

    def update_param(self, eta):
        self.weight -= eta*self.dl_dw_cumulative
        self.bias   -= eta*self.dl_db_cumulative
        
    def zero_grad(self):
        self.dl_dw_cumulative.fill_(0)
        self.dl_db_cumulative.fill_(0)
'''
t+=1
m_t = beta_1*m_t + (1-beta_1)*g_t	#updates the moving averages of the gradient
v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)	#updates the moving averages of the squared gradient
m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
theta_0_prev = theta_0								
theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon) #updates the parameters

for t in range(num_iterations):
    g = compute_gradient(x, y)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    w = w - step_size * m_hat / (np.sqrt(v_hat) + epsilon)
view raw



'''
