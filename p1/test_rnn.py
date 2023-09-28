import torch
from rnn import MyRNN

model = MyRNN(3, 12, 8) #in_dim, hidden_dim, out_dim
x = torch.randn((15, 16, 3)) #batch_size, sequence_length, in_dim

assert(model(x).shape==(15, 16, 8)) #batch_size, sequence_length, out_dim
