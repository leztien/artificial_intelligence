#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bredth first search

AIMA_page_95_breadth_first_search.py
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def make_adjacency_matrix(n, density=None, seed=None):
    """makes a random adjacency matrix representing a graph"""
    p = density or np.log(n) / 10 / n
    
    # Random seed
    if seed is True:
        seed = np.random.randint(1, 9999)
        print("seed =", seed)
    rs = np.random.RandomState(seed)
    
    while True:
        mx = np.triu(rs.rand(n,n))
        mx += mx.T
        mx = (mx <= p).astype("uint8")
        mx[np.diag_indices(n)] = 0
        
        if np.logical_or.reduce(mx, axis=0).all() and np.logical_or.reduce(mx, axis=1).all():
            break
        else:
            p += 0.001
    return(mx)


def display_graph(mx):
    """converts an adjacency matrix into a graph"""
    try:
        G = nx.from_numpy_matrix(mx)
    except AttributeError:
        G = nx.from_numpy_array(mx)
        
    # add coloring to the nodes
    color_map = []
    for node in G:
        if node == 0:
            color_map.append('green')
        elif node == len(mx) - 1:
            color_map.append('red')
        else: 
            color_map.append('cyan')   
            
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()
    
###############################################################################

class Queue:
    """very simple implementation of a FIFO queue based on Python's list"""
    def __init__(self, queue=None): self.queue = list(queue or [])
    def __repr__(self): return str(self.queue)
    def is_empty(self): return len(self.queue) == 0
    def enqueue(self, item): self.queue.append(item)
    def dequeue(self):
        item = self.queue[0]
        del self.queue[0]
        return item
    

@dataclass
class State:
    """atomic state"""
    id: int
    def __hash__(self): return hash(self.id) # to enable hashing (i.e. use in sets)
    def __eq__(self, other): self is other   # to enforce unique states
    def __str__(self): return f"{self.id}"
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.id)
    
    
class Node:
    """Wrapper for a State"""
    def __init__(self, state, parent=None):
        self.state: State = state
        self.parent: Node = parent
    def __str__(self): return "{}({})".format(self.__class__.__name__, self.state)
    def __repr__(self): return self.__str__()


class SearchProblem:
    def __init__(self, problem_representation: 'adjacency matrix'):
        self.adjacency_matrix = problem_representation
        self.statespace = tuple(State(i) for i,_ in enumerate(self.adjacency_matrix))
    
    @property
    def initial(self):
        return self.statespace[0]
    
    def is_goal(self, state):
        assert type(state) is State and state in self.statespace
        return state is self.statespace[-1]
    
    def expand(self, node):
        """
        In AIMA expand() is a stand-alone function, while I have decided to
        make it a method of the SearchProblem() instance, because it is the
        SearchProblem who can know how to expand one of its nodes, not a 
        stand-alone function
        """
        i = node.state.id
        for j in np.nonzero(self.adjacency_matrix[i])[0]:
            yield Node(state=self.statespace[j], parent=node)
    
    

def breadth_first_search(problem):
    """BFS as implemented in AIMA on page 95"""
    
    # wrap the INITIAL state into a node
    node = Node(state=problem.initial)
    
    # if the INITAL state is the GOAL state
    if problem.is_goal(node.state):
        return node
    
    # instantiate the two necessary data structures
    frontier = Queue([node,])   # the queue contains nodes
    reached = {problem.initial} # the set contains states
    
    while not frontier.is_empty():
        node = frontier.dequeue()
        
        for child in problem.expand(node):  # child = child node
            state = child.state
            
            # early goal test
            if problem.is_goal(state):
                return child
            
            if state not in reached:
                reached.add(state)
                frontier.enqueue(child)
    
    # if no solution is found, return failure
    return None # failure
    


if __name__ == '__main__':
    
    seeds = [1244, 7248, 5417, 1642, 9130, 6219, True]
    
    for seed in seeds:
    
        # generate a random adjacency matrix
        mx = make_adjacency_matrix(10, seed=seed)
        
        # display the graph
        display_graph(mx)
        
        # formalize a problem
        problem = SearchProblem(mx)
        
        # BFS
        solution = breadth_first_search(problem)
        print("solution:", solution)
        
        # construct the solution path
        if solution:
            node = solution
            solution_path = [node,]
            
            while not node.state is problem.initial:
                node = node.parent
                solution_path.append(node)
            
            print("solution path:", solution_path[::-1], end='\n\n\n')
    
    
    
