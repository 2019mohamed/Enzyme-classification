# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:06:05 2021

@author: M
"""
from grandiso import find_motifs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch

def automorphism_orbits(graph, print_msgs=False):
    

    # compute the vertex automorphism group
    aut_group = find_motifs(graph, graph)
    #print('iso maps : ',aut_group)
    

    orbit_membership = {}
    for v in graph.nodes():
        orbit_membership[v] = v
    
    
    
    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role
    
    
    orbit_membership_list = [[],[]]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)
    
    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse = True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i,vertex in enumerate(orbit_membership_list[0])}


    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit]+[vertex]
    
    aut_count = len(aut_group)
    
    if print_msgs:
        print('Orbit partition of given substructure: {}'.format(orbit_partition)) 
        print('Number of orbits: {}'.format(len(orbit_partition)))
        print('Automorphism count: {}'.format(aut_count))

    return graph, orbit_partition, orbit_membership, aut_count


def OrbitCountingGivenPattern (g , p , norm = True):
    p, orbit_partition, orbit_membership, aut_count  = automorphism_orbits(p,False)
    
    subgraph_iso = find_motifs(p , g)
    #print(subgraph_iso)
    #feature = [[0 for _ in range(len(orbit_partition)) ] for _ in range(len(g))]
    feature = np.zeros((len(g) , len(orbit_partition)))
    for pattern in subgraph_iso:
        for u , v in pattern.items():
            feature[v][orbit_membership[u]] +=1
    
    return feature/np.max(feature) if norm else feature 
        

def OrbitCountingGivenList (g , l , norm = True):
    Feature = np.zeros((len(g),0))
    for p in l:
        f  = OrbitCountingGivenPattern(g , p , norm = False)
        #print(f)
        Feature = np.concatenate((Feature , f) , axis = 1)
    return Feature/np.max(Feature) if norm else Feature 

'''
p = nx.cycle_graph(4)
print(list(p.edges()))
print(find_motifs(p,p))
print(automorphism_orbits(p, True))



g = nx.gnp_random_graph(6,0.2)
nx.draw_networkx(g)
plt.show()
l = list()
l.append(p)
l.append(p1)
print(OrbitCountingGivenList(g , l , norm = False))
'''