# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:43:54 2021

@author: M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn
import dgl


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.linears[i](h))
            return self.linears[-1](h)

class StrongGIN (nn.Module):
    def __init__(self, input_dim, number_layers , hidden_layers):
        super(StrongGIN,self).__init__()
        self.mlp1 = MLP(num_layers = number_layers, input_dim = input_dim,
                        hidden_dim= input_dim, output_dim = input_dim )
        
        self.mlp2 = MLP(num_layers = number_layers, input_dim = input_dim , 
                        hidden_dim= hidden_layers, output_dim = hidden_layers )
    
    
    def forward(self , graph , feat):
        #graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_u('h', 'm'),
                         self.reduc_func)
        rst = feat + graph.ndata['neigh']
        rst = self.mlp2(rst)
        return rst
        
        
    def reduc_func (self, nodes):
        return {'neigh':self.mlp1(torch.sum(nodes.mailbox['m'] , dim = 1))}
    
class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type,model = 'sgin'):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if model == 'sgin':
                if layer == 0:
                    self.ginlayers.append(StrongGIN(input_dim,num_mlp_layers,hidden_dim))
                else:
                    self.ginlayers.append(StrongGIN(hidden_dim,num_mlp_layers,hidden_dim))
                
            else:
                if layer == 0:
                    mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
                else:
                    mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
    
                self.ginlayers.append(
                    GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            #h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer
    
   
'''
import matplotlib.pyplot as plt

g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
g.ndata['x'] = torch.ones(g.num_nodes(), 3)
print(g)
feat = g.ndata['x']
net = StrongGIN(feat , 1 , 4 , 2)
res = net(g , feat).detach().numpy()
plt.scatter(res[:,0],res[:,1],c = 'r')
for i , l in enumerate(res):
    plt.annotate(i, (l[0],l[1]))
plt.show()
print(res)
lin = nn.Linear(3, 2)
net1 = GINConv(lin, 'max')
res1 = net1(g , feat).detach().numpy()
plt.scatter(res1[:,0],res1[:,1])
for i , l in enumerate(res1):
    plt.annotate(i, (l[0],l[1]))
plt.show()
print(res1)
'''