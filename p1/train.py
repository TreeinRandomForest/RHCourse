import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from rnn import MyRNN, generate_data

# keep all constants at the top - easy to change and run experiments
# some people capitalize them for easy visibility
# in a more robust version, dump constants in a config.py file
# use importlib to read experiment-specific config file specified
# on the command line

# data generation
N = 1000 #num of sequences
L = 20 #length of sequences
low, high = 0, 10

# model dims
in_dim = 1 #treat inputs as floats
hidden_dim = 4 #just picked something - hyperparameter
out_dim = 1 #treat outputs as floats

# generate data
train_data = generate_data(N, L, low=low, high=high)
test_data = generate_data(N, L, low=low, high=high)

# init model
model = MyRNN(in_dim, hidden_dim, out_dim)

# EXERCISE: run print(list(model.parameters())
# and convince yourself that what you see makes sense
optimizer = optim.SGD(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()

def train_loop(model,
               optimizer,
               criterion,
               train_data,
               n_epochs=10
               ):

    model.train() #good practice: will go deeper during dropout and batch normalization

    # training loop begins
    for n in range(n_epochs): #one epoch is one full pass over the train set
        # initialize running sum of training losses
        train_loss = 0 
        for seq in train_data: #loop over every sequence/example
            # step 1: make a prediction
            # seq is a numpy array - convert to torch tensor
            # has shape: (length) -> want (batch size, length, in_dim=1)
            # unsqueeze adds an extra dimension of size 1 at location ARG (0 and 2 in our case)
            seq = torch.from_numpy(seq).unsqueeze(0).unsqueeze(2)

            # cast input tensor to floats
            # weight matrices have float entries
            # computing: matmul(weight, input) -> input type needs to be floats
            seq = seq.float() #cast to floats
            preds = model(seq)

            # preds has shape (1, seq len, 1)
            # there's one output per time index
            # we'll interpret this as a running sum
            
            # step 2: compare to true label
            # we can generate labels from input data
            labels = torch.cumsum(seq, 1) #cumulative sum over dim=1 (seq len)

            # very good habit to add asserts everywhere
            assert labels.shape == seq.shape

            # actually assert here is not that useful since
            # the only transformation we did was the cumsum
            # that, by construction, preserves the shape
            # but if we were to add more code later, this would be
            # a useful sanity check

            # we'll use a simple mean-squared error (MSE) loss
            # average MSEs across tokens/length
            loss = criterion(preds, labels)
            
            # step 3: compute gradients of loss w.r.t. weights (weights, biases)
            # loss.backward() computes gradients
            # the call will **add** gradients to pre-existing buffers
            # this is useful if one is running backprop on parts of a batch
            # when memory is constrained
            # we need to set the buffers to 0 before accumulating gradients
            optimizer.zero_grad() #set buffer values to 0s

            loss.requires_grad = True
            loss.backward() #compute gradients
            
            # step 4: use variant of gradient descent to update weights
            optimizer.step() #super simple
            
            # step 5: evaluate the model on the test set to track performance
            if n % print_freq == 0:
                #run on test set
                #test_acc = test_loop(model, test_data)
                pass
            # update running training loss 
            train_loss += loss.item()

        # step 5: average training loss over all sequences for each epoch
        train_loss /= len(train_data)
        print(f"Epoch {n + 1}/{n_epochs} Loss: {train_loss}")

    # notes:
    # the loop over sequences is inefficient
    # ideally we would pack the data into matrices/tensors and pass to the model
    # will do this but ignore for now

    # the optimizer has internal state just like the model
    # return both so we can do incremental training i.e.
    #
    # model, optimizer = train(model, optimizer, ..., n_epochs=10)
    # model, optimizer = train(model, optimizer, ..., n_epochs=15)

    # should be equivalent to
    # model, optimizer = train(model, optimizer, ..., n_epochs=25) #CHECK
    return model, optimizer

# step 6: evaluate the model on the test set to track performance
def test_loop(model, 
              criterion,
              test_data):
    
    # step 7: set model to evaluation mode
    model.eval()
   
    # initialize running sum of testing losses and accuracies
    test_loss, test_accu = 0, 0

    # step 8: compute gradients without updating weights
    with torch.no_grad():
        for seq in test_data:
            seq = torch.from_numpy(seq).unsqueeze(0).unsqueeze(2)
            seq = seq.float()
            preds = model(seq)
            labels = torch.cumsum(seq, 1)
            
            # update running test loss and accuracy
            test_loss += criterion(preds, labels)
            test_accu += (labels == preds).sum().item()
    
    # step 9: calculate test loss and accuracy over all sequences for each epoch
    test_loss /= len(test_data)
    test_accu /= len(test_data)
    print(f"Test loss: {test_loss} \nTest accuracy: {test_accu}")

    
