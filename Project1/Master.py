'''
    This file automatically trains and runs all networks and collects data about each configuration. 
    It then generates some figures illustrating the performance of the different architectures. 
'''
# Import utility modules
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

    # Train each indiviudal network
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


def eval_all():
    '''
        Eval each model
    '''
    

def main():
    mse_losses, cross_losses = train_all()

    plot_MSEloss(mse_losses)
    plot_CrossEntropy_loss(cross_losses)

    eval_all()

    plt.show() 

if __name__ == "__main__":
    main()