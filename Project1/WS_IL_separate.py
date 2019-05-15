import torch
from torch import Tensor
import dlc_practical_prologue as prologue

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from sub_modules import Parallel_Net, Analyzer_Net 

####################################################################################################
'''
Implementation of 2 independent basic nets for class identification and relationship classifictation:

    - "Weight sharing" (only one network is trained to recognize digits).

    - Intermediate loss (since image recognition is implemented independently from the ANalyzer_Net).

    - They are trained independently, hence the Parallel_Net is not influenced by the loss of the 
      Analyzer_Net. 

We evaluate the collaboration of the 2 networks in 2 distinct ways, by: 
    - Immediatly feeding the output of the Parallel_Net to the Analyzer_Net.
    - Selecting the best class prediction from the Parallel_Net and setting it to 1 while the rest 
      of its output is set to 0, generating a clean input for the Analyzer_Net. 
'''
####################################################################################################

def trainClassIdentifier(model, train_input, train_classes, mini_batch_size):
    
    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the number of epochs to train the network
    epochs = 25
    
    # Set the learning rate
    eta = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr = eta, momentum = 0.0)

    loss_record=[]

    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            model.zero_grad()
            for i in range(2):
                # For each image (since there are 2 channels)
                output = model(train_input[:,i].view(-1,1,14,14).narrow(0, b, mini_batch_size)) # dim,start,length
                loss = criterion(output, train_classes[:,i].narrow(0, b, mini_batch_size))
                sum_loss += loss.item()
                loss.backward()

            optimizer.step()
        print('Sum of loss at epoch {}: \t'.format(e),sum_loss)
        loss_record.append(sum_loss)
    
    return model, loss_record

# Evaluate the network's performance with winner takes it all approach
def evaluateClassIdentification(model, test_input, test_classes, mini_batch_size):
    error = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model(test_input[:,0].view(-1,1,14,14).narrow(0, b, mini_batch_size))
        
        c_array = output.argmax(1)
        t_array = test_classes[:,0][b:b+mini_batch_size]
        error += (c_array-t_array).nonzero().size()[0]
        
    return error/test_input.size()[0]*100


def trainAnalyzer(model, train_classes, train_target, mini_batch_size):
    # Classify into one-hot
    train_target_oneHot = torch.eye(2)[train_target]
    
    # Specify the loss function
    criterion = nn.MSELoss()

    # Define the number of epochs to train the network
    epochs = 25
    
    # Set the learning rate
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = eta, momentum = 0.9)
    loss_record=[]
    
    # One hot encode the training classes and concatenate them (1000x2)->(1000x2x10)
    train_oneHot = torch.empty([train_classes.shape[0],2,10])
    train_oneHot[:,0] = torch.eye(10)[train_classes[:,0]]
    train_oneHot[:,1] = torch.eye(10)[train_classes[:,1]]
    # Convert to the format expected by the Analyzer (1000x20)
    train_oneHot_cat = torch.cat([train_oneHot[:,0],train_oneHot[:,1]],dim=1)

    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_oneHot_cat.size(0), mini_batch_size):
            model.zero_grad()
            output = model(train_oneHot_cat.narrow(0, b, mini_batch_size)) # dim,start,length
            loss = criterion(output, train_target_oneHot.narrow(0, b, mini_batch_size))
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            '''for p in model.parameters():
                p.data.sub_(eta * p.grad.data)'''
        loss_record.append(sum_loss)
        print('Sum of loss at epoch {}: \t'.format(e),sum_loss)
    
    return model, loss_record

def evaluateAnalyzer(model,test_classes, test_target,mini_batch_size):
    # One hot encode the training classes and concatenate them (1000x2)->(1000x2x10)
    test_oneHot = torch.empty([test_classes.shape[0],2,10])
    test_oneHot[:,0] = torch.eye(10)[test_classes[:,0]]
    test_oneHot[:,1] = torch.eye(10)[test_classes[:,1]]
    # Convert to the format expected by the Analyzer (1000x20)
    test_oneHot_cat = torch.cat([test_oneHot[:,0],test_oneHot[:,1]],dim=1)
    
    with torch.no_grad():
        error = 0
        for b in range(0, test_oneHot_cat.size(0), mini_batch_size):
            output = model(test_oneHot_cat.narrow(0, b, mini_batch_size))
            
            for i in range(output.size(0)):
                if torch.argmax(output[i]).item() >= 0.5:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() < 0.2:
                        error += 1
                elif torch.argmax(output[i]).item() < 0.5:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() > 0.8:
                        error += 1
                else:
                    error += 1
    return error/test_target.size(0)*100

def evaluateBothNetworks(parallel_net,analyzer_net,test_input,test_target,mini_batch_size):
    error = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        # Evaluate each image
        output1 = parallel_net(test_input[:,0].view(-1,1,14,14).narrow(0, b, mini_batch_size))
        output2 = parallel_net(test_input[:,1].view(-1,1,14,14).narrow(0, b, mini_batch_size))
        output = torch.cat([output1,output2],dim=1)
        output = analyzer_net(output)

        for i in range(output.size(0)):
                if torch.argmax(output[i]).item() >= 0.5:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() < 0.2:
                        error += 1
                elif torch.argmax(output[i]).item() < 0.5:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() > 0.8:
                        error += 1
                else:
                    error += 1
    return error/test_target.size(0)*100

def evaluateBothNetworksPretty(parallel_net,analyzer_net,test_input,test_target,mini_batch_size):
    error = 0
    for b in range(0, test_input.size(0), mini_batch_size):
        # Evaluate each image with the parallel_net
        output1 = parallel_net(test_input[:,0].view(-1,1,14,14).narrow(0, b, mini_batch_size))
        output2 = parallel_net(test_input[:,1].view(-1,1,14,14).narrow(0, b, mini_batch_size))

        # Prettify the output of the first network
        output1 = torch.eye(10)[torch.argmax(output1,dim=1)]
        output2 = torch.eye(10)[torch.argmax(output2,dim=1)]
        
        # Evaluate the analyzer_net 
        output = torch.cat([output1,output2],dim=1)
        output = analyzer_net(output)

        for i in range(output.size(0)):
                if torch.argmax(output[i]).item() >= 0.5:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() < 0.2:
                        error += 1
                elif torch.argmax(output[i]).item() < 0.5:
                    if test_target.narrow(0, b, mini_batch_size)[i].item() > 0.8:
                        error += 1
                else:
                    error += 1
    return error/test_target.size(0)*100

def Net():
    return Parallel_Net(), Analyzer_Net()

def main():
    '''
        Setup both networks and train them individually
    '''
    # Load the dataset
    nb = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

    # Define the mini_batch size
    mini_batch_size = 100

    # Declare both instances of the nets
    parallel_net, analyser_net = Net()

    # Train both networks
    parallel_net, _ = trainClassIdentifier(parallel_net, train_input, train_classes, mini_batch_size)
    analyser_net, _ = trainAnalyzer(analyser_net, train_classes, train_target, mini_batch_size)

    '''
        Evaluate both networks independently
    '''
    # Evaluate the class identification of the model
    res = evaluateClassIdentification(parallel_net, test_input, test_classes, mini_batch_size)
    print('Error rate of the parallel model: ',res,'%')
    # Evaluate the relation prediction between the digit of the model
    res = evaluateAnalyzer(analyser_net,test_classes,test_target,mini_batch_size)
    print('Error rate of the analyzer model: ',res,'%') 

    '''
        Evaluate both network together by feeding directly the output of the first net to the second
    '''
    # Evaluate the combined performance of both networks
    res = evaluateBothNetworks(parallel_net,analyser_net,test_input,test_target,mini_batch_size)
    print('Error rate of the combined models: ',res,'%')

    '''
        Before feeding the output of the first net to the second, make the output prettier by setting
        the max class to 1 and all the other to 0 to reduce the error of the analyzer. 
    '''
    res = evaluateBothNetworksPretty(parallel_net,analyser_net,test_input,test_target,mini_batch_size)
    print('Error rate of the combined models with pretty intermediate values: ',res,'%')


if __name__ == "__main__":
    main()