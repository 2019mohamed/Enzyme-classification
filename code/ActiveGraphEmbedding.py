# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:36:43 2021

@author: M
"""

from karateclub import FeatherGraph
import modAL
from dgl.data import GINDataset
from GraphModels import GIN
from GraphDataLoader import GDataLoader
import dgl
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neural_network import MLPClassifier
import torch
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

len_emd = 500
def DGLtoemd (loader):
    data , y = [],[]
    embedder = FeatherGraph()

    for g , l in loader:
        g = g.to_networkx().to_undirected()
        g = nx.Graph(g)
        nx.draw_networkx(g)
        plt.show()
        l = l.detach().numpy()[0]
        if nx.is_connected(g):
            embedder.fit([g])
            data.append(embedder.get_embedding()[0])
            y.append(l)       
        else:
            CC = nx.connected_components(g)
            embCC = []
            for c in list(CC):
                relabel = { n : i for i,n in enumerate(c)}
                sub = nx.subgraph(g,nbunch = c)
                sub = nx.relabel_nodes(sub , relabel)                
                embedder.fit([sub])
                embCC.append(embedder.get_embedding()[0])
            
            embCC = np.array(embCC)
            embCC = np.concatenate(tuple(embCC) , axis = 0)
            embCC = np.random.choice(embCC, size=500, replace=False)
            data.append(embCC)
            y.append(l)
    
    return np.array(data) , np.array(y)
    

data = GINDataset(name='PROTEINS', self_loop=True)
Loader = GDataLoader(data,1 , collate)
train_loader , test_loader = Loader.train_valid_loader()


test_emd , test_y = DGLtoemd(test_loader)

train_emd , train_y = DGLtoemd(train_loader)
print('all size data ',len(train_emd) + len(test_emd))
print('Data done')

initial_idx = np.random.choice(range(len(train_emd)), size=50, replace=False)
X_init = train_emd[initial_idx]
y_init = train_y[initial_idx]

X_pool = np.delete(train_emd, initial_idx, axis=0)
y_pool = np.delete(train_y, initial_idx, axis=0)

clf = MLPClassifier(random_state=42, max_iter=2000)
#clf.fit(train_emd , train_y)
#print(clf.score(test_emd , test_y))

learner = ActiveLearner(
    estimator=clf,
    query_strategy=uncertainty_sampling,
     X_training=X_init , y_training=y_init
)

print(learner.score(test_emd, test_y))

n_queries = 20
for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool, n_instances=20)
    learner.teach(X_pool[query_idx], y_pool[query_idx], only_new=False)
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    print(learner.score(test_emd, test_y))


# the final accuracy score
print('Final ',learner.score(test_emd, test_y))






   

