#%%
class DLModule(object):   
    def  forward(self , *input):
        raise  NotImplementedError

    def  backward(self , *gradwrtoutput):
        raise  NotImplementedError

    def  param(self):
        return  []

#
class tanh(object):
    
    def eval(self, x):
        return x

    def evald(self, x):
        return dx



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
        self.function = activation_function
        self.weight = torch.empty((input_dim, output_dim))
        self.bias = torch.empty(1, output_dim)
        self.grad_weight = torch.empty(output_dim, input_dim)
        self.grad_bias = torch.empty(1, output_dim)
    
    def forward(self, x):
        self.input_previous = x
        self.s = w*x+b 
        return f(self.s)

    def backward(self, dl_dx):
        '''
        dl_dx received
        compute dl_ds = dl_dx .*dF(self.s)
        grad_weight dl_dw_cumulative += dl_ds*self.input ( input from forward pass)
        grad_bias dl_db_cumulative += dl_ds
        generate dl_dx_(l-1) for next layer : weight*dl_ds


        '''
        dl_ds_ = dl_dx_received * 

        return dl_dx_(l-1)

    def update_param():
        
    def zero_grad():


#%%
