import torch
import torch.nn as nn #high-level neural network classes and functions
import torch.optim as optim #optimization (sgd and variants) functions

import numpy as np

def generate_data(N, L, low=0, high=10):
    '''
    Generate N sequences of integers between low and high
    Each sequence has length L
    '''

    return np.random.randint(low=low, high=high, size=(N, L), dtype=int)

class MyRNN(nn.Module): #your model will inherit from nn.Module
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 out_dim):
        
        super().__init__() #always call parent class' constructor

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.act = nn.ReLU()

        # initialize weights - see torch.nn.init.*
        # in ipython (or jupyter) - type: torch.nn.init.*?
        # see: https://cs230.stanford.edu/section/4/
        # will see better initialization in next section: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self.W_hx = nn.Parameter(torch.randn(size=(hidden_dim, input_dim))) #read as mapping x space -> h space
        self.W_hh = nn.Parameter(torch.randn(size=(hidden_dim, hidden_dim))) #read as mapping h space -> h space
        self.b_h = nn.Parameter(torch.zeros(size=(hidden_dim,))) #bias in h space

        self.W_yh = nn.Parameter(torch.randn(size=(out_dim, hidden_dim))) #mapping h space -> y space
        self.b_y = nn.Parameter(torch.zeros(size=(out_dim,))) #bias in y space
        #self.b_y = torch.zeros(size=(out_dim,)) #bias in y space

        # note: as we'll see later, pytorch needs to treat
        # weights specially i.e. they need to be registered
        # so autodiff/backprop keeps track of their derivatives
        # this can be done by wrapping them in an nn.Parameter call
        # or putting them in a parameter list
        # Exercise: comment the following line, init the model
        # and look at list(model.parameters())
        # Now do the same after uncommenting the following line
        self.params = nn.ParameterList([self.W_hx,
                                        self.W_hh,
                                        self.b_h,
                                        self.W_yh,
                                        self.b_y
                                        ])
        
    def forward(self, x): #forward function always defines forward propagation
        #recall loop defined in readme file

        num_seqs = x.shape[0]
        len_seqs = x.shape[1]
        #can write: num_seqs, len_seqs = x.shape
        
        hidden = torch.zeros(size=(num_seqs, self.hidden_dim))
        # special note: we are assuming above that all the input seqs
        # in x have the same length. This is generally not true but we
        # won't cover the general case here

        # understanding mat mul in pytorch
        # a = torch.randn(3, 4) #a.shape - (3, 4) - think of this as the weights matrix
        # b = torch.randn(15, 10, 4) #b.shape - (15, 10, 4) - think of this as (batch, time, embedding)
        # we want to multiply a by every 4-dim vector, at a certain time-step (second index)
        # for all sequences in the batch (15)
        #
        # see doc for torch.matmul - https://pytorch.org/docs/stable/generated/torch.matmul.html - specifically the rules for broadcasting (5th bullet point)
        # we want: torch.matmul(a, b[:, 0, :]).shape == (15, 1, 3)
        # b.shape == (15, 10, 4)
        # b[:,0,:].shape == (15, 4)
        # b[:,0,:].unsqueeze(2).shape == (15, 4, 1) [unsqueeze adds a wrapping dimension at the argument index]
        # torch.matmul(a, b[:,0,:].unsqueeze(2)).shape == (15,3,1)
        # Summary:
        #     matmul(a, b[:,0,:].unsqueeze(2))
        #     matmul(shape == (3,4), shape == (15,4,1)) -> shape==(15,3,1)
        
        
        out_array = torch.zeros(size=(len_seqs, num_seqs, self.out_dim))
        for i in range(len_seqs):
            # exercise: carefully experiment with the line below
            # is it doing the correct computation?
            # hint: create random vectors and examine shapes
            # hint: use ipdb as shown below to set breakpoint
            # import ipdb
            # ipdb.set_trace()
            hidden = self.act(torch.matmul(self.W_hx, x[:, i, :].unsqueeze(2)).squeeze(2) +\
                              torch.matmul(self.W_hh, hidden.unsqueeze(2)).squeeze(2) +\
                              self.b_h)
            
            out = torch.matmul(self.W_yh, hidden.unsqueeze(2)).squeeze(2) + self.b_y
            out_array[i, :, :] = out

        return out_array.permute(1,0,2)

                               
                               
