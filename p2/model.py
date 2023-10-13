import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 n_input,
                 n_output,
                 n_hidden_layers,
                 n_hidden_nodes,
                 act):
        super().__init__()

        self.layer_list = nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        
        for i in range(n_hidden_layers):
            if i==0:            
                l = nn.Linear(n_input, n_hidden_nodes)
            else:
                l = nn.Linear(n_hidden_nodes, n_hidden_nodes)

            self.layer_list.append(l)

        l = nn.Linear(n_hidden_nodes, n_output)
        self.layer_list.append(l)

        assert(len(self.layer_list) == n_hidden_layers+1)
        
        self.act = act

    def forward(self, x):
        out = x
        for idx, l in enumerate(self.layer_list):
            if idx != self.n_hidden_layers:
                out = self.act(l(out))
            else:
                out = l(out)

        return out
