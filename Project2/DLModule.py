#%%
class DLModule(object):  

    def __init__(self, args):

    def sequential(self, *args):
        self.modlist.append(*args)   

    def  forward(self , input):
        for node in self.mod_list:
            input = node.forward(input)
        return input
    
    def  backward(self , output):
        for node in self.mod_list:
            output = node.backward(output)   
    def update(self,eta): 
        for node in nodelineraires:
            node.update_param(eta)

        

class Loss(object):
    def MSE(self, output, target):
        return torch.sum(torch.pow(output-target,2))

    def dMSE(self, output, target):
        return 2*(output-target)    

class tanh(object):
    
    def eval(self, x):
        return tanh(x)

    def evald(self, x):
        return d(tanh(x))

    def forward(self, input):
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        dl_ds = dl_dx * self.evald(self.input)    

class relu(object):
    
    def eval(self, x):
        return relu(x)

    def evald(self, x):
        return d(relu(x))

    def forward(self, input):
        self.input = input    
        return self.eval(input)

    def backward(self, dl_dx):
        dl_ds = dl_dx * self.evald(self.input)




class Linear(object):
'''
    One fully connected layer
    batch of size [n_sample x input_dim]

    after eval size [n_sample x output_dim]
    
'''
    def __init__(self, activation_function, input_dim, output_dim):
        '''
            weight size [input_dim x output_dim]
            bias size [ 1 x output_dim]
            grad_weight [ input_dim x output_dim]
            grad_bias [ 1 x output_dim]
        '''
        self.weight = torch.empty((input_dim, output_dim))
        self.bias = torch.empty(1, output_dim)
        self.grad_weight = torch.empty(output_dim, input_dim)
        self.grad_bias = torch.empty(1, output_dim)
    
    def forward(self, x):
        self.input_previous = x
        return w*x+b

    def backward(self, dl_ds):
        '''
        dl_dx received
        compute dl_ds = dl_dx .*dF(self.s)
        grad_weight dl_dw_cumulative += dl_ds*self.input ( input from forward pass)
        grad_bias dl_db_cumulative += dl_ds
        generate dl_dx_(l-1) for next layer : weight*dl_ds
        '''
        dl_dx = wt*dl_ds
        self.dl_dw_cumulative += dl_ds * self.input_previous
        self.dl_db_cumulative += dl_ds

        return dl_dx

    def update_param(eta):
        w -= eta*self.dl_dw_cumulative
        b -= eta*self.dl_db_cumulative
        
    def zero_grad():
        self.dl_dw_cumulative=torch.empty(output_dim, input_dim)
        self.dl_db_cumulative=torch.empty(1, output_dim)


#%%

