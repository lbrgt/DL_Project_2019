'''
    This file automatically trains and runs all networks and collects data about each configuration. 
    It then generates some figures illustrating the performance of the different architectures. 
    
    usage: Master.py [-h] [-t] [-s] [-l]
    optional arguments:
        -h, --help     show this help message and exit
        -t, --trained  use pretrained networks
        -s, --stat     train and evaluate 10 times the networks and return their error rates
        -l, --load     load precomputed statistics (use it with -s option)
'''
# Import utility modules
import argparse
import pickle 
import numpy as np 
from matplotlib import pyplot as plt
import dlc_practical_prologue as prologue

# Import all architectures
import NoWS_NoIL        # No weight sharing, no intermediate loss
import NoWS_IL          # No weight sharing, intermediate loss
import WS_NoIL          # Weight sharing, no intermediate loss
import WS_IL            # Weight sharing, intermediate loss
import WS_IL_separate   # Weight sharing, intermediate loss, ...
                        #   ... but the two parts of the network are trained completely separately

# Load the dataset
nb = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)

# Define the minibatch size
mini_batch_size = 200

# Instantiate each network
nows_noil_model = NoWS_NoIL.Net()
nows_il_model   = NoWS_IL.Net()
ws_noil_model   = WS_NoIL.Net()
ws_il_model     = WS_IL.Net()
ws_il_sep_model_parallel, ws_il_sep_model_analyzer = WS_IL_separate.Net() 


def plot_MSEloss(mse_losses:list):
    '''
        Plot the MSE losses of each Network
    '''
    # Create a new figure
    _ = plt.figure() 
    # Plot the MSE Loss
    plt.plot(mse_losses[0],label='nows_noil')
    plt.plot(mse_losses[1],label='nows_il')
    plt.plot(mse_losses[2],label='ws_noil')
    plt.plot(mse_losses[3],label='ws_il')
    plt.plot(mse_losses[4],label='separated')

    # Finalize the figure
    plt.title('MSE loss for different configurations')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss over minibatch of size {}'.format(mini_batch_size))
    plt.grid(True)
    plt.legend() 
    plt.tight_layout()
    plt.savefig('figures/MSELoss.png')

def plot_CrossEntropy_loss(cross_losses:list):
    '''
        Plot the CrossEntropy losses of each Network
    '''
    # Create a new figure
    _ = plt.figure() 
    # Plot the MSE Loss     nows_il_loss_classifier,ws_il_loss_classifier,sep_loss_parallel
    plt.plot(cross_losses[0],label='nows_il')
    plt.plot(cross_losses[1],label='ws_il')
    plt.plot(cross_losses[2],label='separated')

    # Finalize the figure
    plt.title('Cross Entropy loss for different configurations')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss over minibatch of size {}'.format(mini_batch_size))
    plt.grid(True)
    plt.legend() 
    plt.tight_layout()
    plt.savefig('figures/CELoss.png')

def train_all():
    '''
        Train each network with the provided samples
    '''
    global nows_noil_model,nows_il_model,ws_noil_model,ws_il_model,\
        ws_il_sep_model_parallel,ws_il_sep_model_analyzer

    # Train each individual network
    # I - NoWS_NoIL
    print('Training NoWS_NoIL:')
    nows_noil_model, nows_noil_loss = NoWS_NoIL.train_network(nows_noil_model, train_input, train_target, mini_batch_size)
    # II - NoWS_IL
    print('Training NoWS_IL:')
    nows_il_model, nows_il_loss, nows_il_loss_classifier = NoWS_IL.train_network(
        nows_il_model, train_input, train_target, train_classes, mini_batch_size)
    # III - WS_NoIL
    print('Training WS_NoIL:')
    ws_noil_model, ws_noil_loss = WS_NoIL.train_network(ws_noil_model, train_input, train_target, mini_batch_size)
    # IV - WS_IL
    print('Training WS_IL:')
    ws_il_model, ws_il_loss, ws_il_loss_classifier = WS_IL.train_network(
        ws_il_model, train_input, train_target, train_classes, mini_batch_size)
    # V - WS_IL_separate
    print('Training WS_IL_separate:')
    ws_il_sep_model_parallel, sep_loss_parallel = WS_IL_separate.trainClassIdentifier(
        ws_il_sep_model_parallel, train_input, train_classes, mini_batch_size)
    ws_il_sep_model_analyzer, sep_loss_analyzer = WS_IL_separate.trainAnalyzer(
        ws_il_sep_model_analyzer, train_classes, train_target, mini_batch_size)

    print('Training done')

    mse_losses = [nows_noil_loss,nows_il_loss,ws_noil_loss,ws_il_loss,sep_loss_analyzer]
    cross_losses = [nows_il_loss_classifier,ws_il_loss_classifier,sep_loss_parallel]
    return mse_losses,cross_losses

def plot_result(res_class:dict, res_final:dict):
    #_ = plt.figure()
    #plt.plot(res_class.items())

    _ = plt.figure()
    for i in res_final.items():
        plt.plot(i[0],i[1],color='b',marker='o')

    # Finalize the figure
    plt.title('Error rate of the different architectures')
    plt.xlabel('Architecture type')
    plt.ylabel('Error rate in %')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/ErrorRate.png')

def eval_all():
    '''
        Eval each model
    '''
    # I - NoWS_NoIL
    nows_noil_res = NoWS_NoIL.evaluateFinalOutput(nows_noil_model,test_input,test_target,mini_batch_size)

    # II - NoWS_IL
    #nows_il_res_class = NoWS_IL.evaluateClassIdentification(nows_il_model, test_input, test_classes, mini_batch_size)
    nows_il_res = NoWS_IL.evaluateFinalOutput(nows_il_model, test_input, test_target, mini_batch_size) 

    # III - WS_NoIL
    ws_noil_res = WS_NoIL.evaluateFinalOutput(ws_noil_model, test_input, test_target, mini_batch_size)

    # IV - WS_IL
    #ws_il_res_class = WS_IL.evaluateClassIdentification(ws_il_model, test_input, test_classes, mini_batch_size)
    ws_il_res = WS_IL.evaluateFinalOutput(ws_il_model, test_input, test_target, mini_batch_size)
    
    # V - WS_IL_separate
    ws_il_sep_res_class = WS_IL_separate.evaluateClassIdentification(
        ws_il_sep_model_parallel, test_input, test_classes, mini_batch_size)
    ws_il_sep_res = WS_IL_separate.evaluateBothNetworks(
        ws_il_sep_model_parallel, ws_il_sep_model_analyzer, test_input, test_target, mini_batch_size)
    ws_il_sep_res_pretty = WS_IL_separate.evaluateBothNetworksPretty(
        ws_il_sep_model_parallel, ws_il_sep_model_analyzer, test_input, test_target, mini_batch_size)

    # Return a dictionary of the performances:
    res_class = {
        #'nows_il':nows_il_res_class,
        #'ws_il':ws_il_res_class,
        'ws_il_sep':ws_il_sep_res_class
    }
    res_final= {
        'nows_noil':nows_noil_res,
        'nows_il':nows_il_res,
        'ws_noil':ws_noil_res,
        'ws_il':ws_il_res,
        'ws_il_sep':ws_il_sep_res,
        'ws_il_pretty':ws_il_sep_res_pretty
    }
    return res_class, res_final

def plot_statistics(keys:np.array,stats:np.array):
    _ = plt.figure()
    plt.boxplot(stats) 
    plt.title('Error rate statistics over 10 runs')
    xlabel = '        '
    for k in keys:
        xlabel += str(k) + '        '
    plt.xlabel(xlabel + '\nArchitecture type')
    plt.ylabel('Error rate in %')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/ErrorRateStat.png')

def generate_statistics():
    global nows_noil_model,nows_il_model,ws_noil_model,ws_il_model,\
        ws_il_sep_model_parallel,ws_il_sep_model_analyzer

    global train_input, train_target, train_classes, test_input, test_target, test_classes
    
    res_final_stat = []
    for i in range(10):
        # Reintialize each network
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)
        nows_noil_model = NoWS_NoIL.Net()
        nows_il_model   = NoWS_IL.Net()
        ws_noil_model   = WS_NoIL.Net()
        ws_il_model     = WS_IL.Net()
        ws_il_sep_model_parallel, ws_il_sep_model_analyzer = WS_IL_separate.Net() 

        _, _ = train_all()
        _, res_final = eval_all() 
        res_keys = list(res_final.keys()) 
        res_values = list(res_final.values())
        res_final_stat.append(res_values)

    res_keys = np.array(res_keys)
    res_final_stat = np.array(res_final_stat)
    res_final_stat = res_final_stat 
    outfile = open('pickle/stats','wb')
    pickle.dump([res_keys,res_final_stat], outfile) 
    outfile.close() 
    return np.array(res_keys), res_final_stat

def computeErrorRateSTD(keys:np.array, stats:np.array):
    # stats is (10,6) array -> compute std along columns
    stds  = np.std(stats,axis=0)
    #print('Standard deviation of each architecture:',stds)

    plt.figure()
    plt.scatter(keys,stds)
    plt.title('Standard deviation of the error rate of the different architectures')
    plt.xlabel('Architecture type')
    plt.ylabel('Standard deviation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/stds.png')

def main(parser:argparse.ArgumentParser):
    global nows_noil_model,nows_il_model,ws_noil_model,ws_il_model,\
        ws_il_sep_model_parallel,ws_il_sep_model_analyzer
    
    # Get the command line arguments
    args = parser.parse_args()

    # If the script is run in stat mode
    if args.statistic:
        if not args.load:
            res_final_keys, res_final_stat = generate_statistics()
        else:
            # Load the stored losses
            infile = open('pickle/stats','rb')
            stats = pickle.load(infile)
            res_final_keys, res_final_stat = stats[0], stats[1]
            infile.close()

        plot_statistics(res_final_keys, res_final_stat) 
        computeErrorRateSTD(res_final_keys, res_final_stat)
        
        plt.show() 
        return 

    # Get or generate the networks and losses
    if args.pretrained:
        print('Using pretrained networks')
        # Load the stored losses
        infile = open('pickle/losses','rb')
        losses = pickle.load(infile)
        mse_losses, cross_losses = losses[0], losses[1]
        infile.close()
        # Load the trained networks
        infile = open('pickle/trainedNet','rb')
        nets = pickle.load(infile)
        nows_noil_model = nets[0]
        nows_il_model = nets[1]
        ws_noil_model = nets[2]
        ws_il_model = nets[3]
        ws_il_sep_model_parallel = nets[4]
        ws_il_sep_model_analyzer = nets[5] 
        infile.close()
    else:
        # Train the networks
        mse_losses, cross_losses = train_all()
        # Dump the recorded losses
        outfile = open('pickle/losses','wb')
        pickle.dump([mse_losses, cross_losses], outfile) 
        outfile.close() 
        # Dump the trained networks
        outfile = open('pickle/trainedNet','wb')
        pickle.dump([nows_noil_model,
                    nows_il_model,
                    ws_noil_model,
                    ws_il_model,
                    ws_il_sep_model_parallel,
                    ws_il_sep_model_analyzer], outfile) 
        outfile.close() 

    plot_MSEloss(mse_losses)
    plot_CrossEntropy_loss(cross_losses)


    res_class, res_final = eval_all()
    plot_result(res_class, res_final)

    # Display all generated figures
    plt.show() 

if __name__ == "__main__":
    # Setup the command line parser
    parser = argparse.ArgumentParser(description='Master script to train and run predefined networks')
    parser.add_argument('-t','--trained', action='store_true', default=False,
                    dest='pretrained',
                    help='use pretrained networks')
    parser.add_argument('-s','--stat', action='store_true', default=False,
                    dest='statistic',
                    help='train and evaluate 10 times the networks and return their error rates')
    parser.add_argument('-l','--load', action='store_true', default=False,
                    dest='load',
                    help='load precomputed statistics (use it with -s option)')
    main(parser) 