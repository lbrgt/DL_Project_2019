#%%
class NeuralNet(object):
    class DLModule(object):  

        def __init__(self, args):
            self.wsh = 1

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

            

    class LossMSE(object):
        def eval(self, output, target):
            return torch.sum(torch.pow(output-target,2))

        def evald(self, output, target):
            return 2*(output-target)    

    class Tanh(object):
        
        def eval(self, x):
            return x.tanh()

        def evald(self, x):
            return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

        def forward(self, input):
            self.input = input    
            return self.eval(input)

        def backward(self, dl_dx):
            return dl_dx * self.evald(self.input)    

    class Relu():
        def eval(self, x):
            return x.relu()
        
        def evald(self, x):
            return (x>0).type(torch.FloatTensor)

        def forward(self, input):
            self.input = input    
            return self.eval(input)

        def backward(self, dl_dx):
            return dl_dx * self.evald(self.input)
    
    class Linear(object):
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
            self.weight = torch.empty((input_dim, output_dim)).fill_(1)
            self.bias = torch.empty(1, output_dim).fill_(1)
            self.dl_dw_cumulative = torch.empty(input_dim, output_dim)
            self.dl_db_cumulative = torch.empty(1, output_dim)
        
        def forward(self, x):
            self.input_previous = x
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

        def update_param(eta):
            w -= eta*self.dl_dw_cumulative
            b -= eta*self.dl_db_cumulative
            
        def zero_grad():
            self.dl_dw_cumulative=torch.empty(output_dim, input_dim)
            self.dl_db_cumulative=torch.empty(1, output_dim)


    #%%



#%%
import NeuralNet as nn


#%%

NeuralNet.relu.eval(a)

#%%
# a =torch.Tensor([ -10, 20, 30])


#%%
