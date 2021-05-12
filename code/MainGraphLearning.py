# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:57:30 2021

@author: M
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import GINDataset
from GraphModels import GIN
from GraphDataLoader import GDataLoader
import dgl
import matplotlib.pyplot as plt
import networkx as nx
from Orbit_Counting import OrbitCountingGivenList

#subgraph_counts_train , subgraph_counts_test = [] , []

#List_patterns = [nx.cycle_graph(3) , nx.cycle_graph(4)]


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels



def train(net, trainloader, optimizer, criterion, epoch):
    net.train()

    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    #bar = tqdm(range(total_iters))

    for idx , data  in enumerate(trainloader):
        # batch graphs will be shipped to device in forward part of model
        #labels = labels.to(args.device)
        #graphs = graphs.to(args.device)
        graphs, labels = data
        feat = graphs.ndata['attr']
        feat = torch.tensor(np.random.standard_normal(size=(len(feat), 2)))
        #feat  = torch.tensor(OrbitCountingGivenList(graphs.to_networkx().to_undirected()
         #                                          , List_patterns))
        #print(feat.shape)
        #print(feat)
        outputs = net(graphs, feat.float())

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        #bar.set_description('epoch-{}'.format(epoch))
    #bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(net, dataloader, criterion):
    net.eval()

    total = 0
    total_loss = 0
    total_correct = 0

    for idx ,  data in enumerate(dataloader):
        graphs, labels = data
        feat = graphs.ndata['attr']
        #feat = torch.tensor(OrbitCountingGivenList(graphs.to_networkx().to_undirected()
         #                                          , List_patterns))
        feat = torch.tensor(np.random.standard_normal(size=(len(feat), 2)))
        #feat = subgraph_counts_test[idx]
        #feat= torch.tensor(feat)
        total += len(labels)
        outputs = net(graphs, feat.float())
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

    loss, acc = 1.0*total_loss / total, 1.0*total_correct / total

    #net.train()

    return loss, acc

data = GINDataset(name='PROTEINS', self_loop=True)
Loader = GDataLoader(data,16 , collate)

train_data , test_data = Loader.train_valid_loader()

init_data = Loader.GW_method(collate, 64)

model = GIN(
    num_layers=4 , num_mlp_layers=3 ,input_dim= 2 , 
    hidden_dim = 128 , output_dim = 2 , final_dropout=0.5 ,
    learn_eps = False ,neighbor_pooling_type= 'sum' ,graph_pooling_type= 'sum',model = 'sgin')

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def train_sample (init_data):
    LT , LS , Ac= [] , [], []
    e = 500
    for epoch in range(e):
            loss_train = train(model, init_data, optimizer, criterion, epoch)
            LT.append(loss_train)
            scheduler.step()
            test_loss, test_acc = eval_net(model, test_data, criterion)
            LS.append(test_loss)
            Ac.append(test_acc)
            if epoch % 10 == 0 or epoch == 499:
                print(epoch,' train ',loss_train)
                print('test ',test_loss,'  ' ,test_acc)
    
    plt.plot(range(len(LT)) , LT , c = 'red' , label = 'train loss')
    plt.plot(range(len(LT)) , LS , label = 'test loss')
    #plt.plot(range(len(LT)) , Ac , c = 'black' , label ='test accurcy' )
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

#train_sample(train_data)

print(len(Loader.train_idx))
train_sample(init_data)
for i in range(2):
    init_data = Loader.GW_method(collate , 64)
    print(len(Loader.train_idx))
    train_sample(init_data)
