#!/usr/bin/env python3

"""
BFS vs DFS
"""



from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import NamedTuple, Union, Optional, Any
from functools import reduce
from operator import or_
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



##### Utilities to generate a random graph and the display thtereof #####

def generate_graph(n, seed=None):
    """
    Generate a random prahp in this form:
        graph = {'vertices': {A, B, C, D}, 'edges': {{A,B}, {B,C}}}
    """
    
    # limitations
    if n > 26:
        raise ValueError("n must be less that or equal tomthe nmber of the letters innthe latin alphabet")
    
    # initialize a graph object
    graph= dict(vertices={chr(65+i) for i in range(n)}, edges=set())
    
    # Random seed
    if seed is True:
        seed = np.random.randint(1, 9999)
        print("seed =", seed)
    rs = np.random.RandomState(seed)
    
    # make a random adj mx
    p = np.log(n) / 10 / n
    while True:
        mx = np.triu(rs.rand(n,n))
        mx += mx.T
        mx = (mx <= p).astype("uint8")
        mx[np.diag_indices(n)] = 0
        
        if mx.sum(axis=0).min() > 0: break
        else: p += 0.001
            
    # convert the adj mx into edges dict
    for i,row in enumerate(np.triu(mx)[:-1]):
        for j in np.nonzero(row)[0]:
            graph['edges'].add(frozenset({chr(65+i), chr(65+j)}))
    return graph
            

def display_graph(graph):
    """
    Vizualize a graph passed in in this form:
    graph = {'vertices': {A, B, C, D}, 'edges': {{A,B}, {B,C}}}
    """
    plt.figure()
    G = nx.Graph()
    G.add_nodes_from(graph['vertices'])
    G.add_edges_from(graph['edges'])
        
    # add coloring to the nodes
    color_map = []
    for node in G:
        if node == min(graph['vertices']):
            color_map.append('green')
        elif node == max(graph['vertices']):
            color_map.append('red')
        else: 
            color_map.append('cyan')
    
    nx.draw(G, with_labels=True, node_color=color_map,)
    plt.show()


   
#### Data Structures to fomralize the problem #####

class AbstractQueue(ABC):
    """Abstract Queue for FIFO and LIFO queues"""
    def __init__(self, items=None): self._items = list(items or [])
    def __repr__(self): return f"{self.__class__.__name__}({str(self._items)[1:-1]})"
    def __getitem__(self, index): return self._items[index]
    def __delitem__(self, index): del self._items[index]
    def __len__(self): return len(self._items)
    def __bool__(self): return len(self._items) > 0
    def is_empty(self): return len(self) == 0
    def add(self, item): self._items.append(item)
    @abstractmethod
    def pop(self):
        "pop-right for stack bzw pop-left for queue"
        if len(self) == 0: raise IndexError("The queue is empty")


class Queue(AbstractQueue):
    """FIFO queue"""
    def pop(self):
        super().pop()
        item = self[0]
        del self[0]
        return item


class Stack(AbstractQueue):
    """LIFO queue i.e. stack"""
    def pop(self):
        super().pop()
        return self._items.pop()


@dataclass
class Node:
    state: Union[int, str, Any]
    parent: Optional['Node'] = None
    action: Optional[Any] = None
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.state)


class AbstractSearchProblem(ABC):
    """
    Abstract Search Problem class
    """
    def __init__(self, problem_representation=None):
        self.problem_representation = problem_representation
    
    @abstractproperty
    def initial(self):
        "returns the INITIAL state"
    
    @abstractmethod
    def is_goal(self, state) -> bool:
        "returns True if the state is a goal state"
    
    @abstractmethod
    def actions(self, state):
        "returns valid actions from the given state"
    
    @abstractmethod
    def result(self, state, action):
        "transition model"
    
    def action_cost(self, state, action, next_state) -> float:
        "Returns the cost of performing an action from the current state that takes the agent to the next_state"
        "Not needed in BFS and DFS"
        return 1.0


class SearchProblem(AbstractSearchProblem):
    """
    Concrete search problem representation and implementation.
    """
    def __init__(self, problem_representation):
        """
        Expects a graph as problem representation, in this format:
        graph = {'vertices': {A, B, C, D}, 'edges': {{A,B}, {B,C}}}
        """
        self.graph = problem_representation

    @property
    def initial(self):
        """Returns the INITIAL state"""
        return min(self.graph['vertices'])
    
    def is_goal(self, state) -> bool:
        return state == max(self.graph['vertices'])
    
    def actions(self, state) -> set:
        return reduce(or_, {edge for edge in self.graph['edges'] if state in edge}, frozenset()).difference({state})
        
    def result(self, state, action):
        """transition model"""
        next_state = action  # here
        edge = frozenset({state, action})
        return next_state if edge in self.graph['edges'] else None
    
    def expand(self, node):
        """
        note: in AIMA, expand() is a stand-alone function
        """
        state = node.state
        for action in self.actions(state):
            next_state = self.result(state, action)
            yield Node(state=next_state, parent=node)
    
    def get_solution_path(self, node):
        assert self.is_goal(node.state), "node must be the GOAL"
        path = [node.state,]
        while node.state != self.initial:
            node = node.parent
            path.append(node.state)
        return path[::-1]

    

##### The most important part - the search functions themselves #####

def search(problem, search_type="BFS"):
    """
    BFS vs DFS
    """
    node = Node(problem.initial)
    
    if problem.is_goal(node.state):
        return node
    
    frontier = Queue([node,]) if str(search_type).upper()=='BFS' else Stack([node,])
    reached = {problem.initial}  # a set of states
    
    while frontier:
        node = frontier.pop()
        for child in problem.expand(node):
            state = child.state
            if problem.is_goal(state):
                return child
            if state not in reached:
                reached.add(state)
                frontier.add(child)

    # return failure
    return None


def _recurse(node, problem, reached):
    """The recurser function for the recursive depth first search"""
    # base case
    if problem.is_goal(node.state):
        return node
    
    # recursive case
    reached.add(node.state)
    
    for child in problem.expand(node):
        if child.state in reached:
            continue
        
        output = _recurse(child, problem, reached)
        if output is not None:
            return output
        
    # return failure
    return None


def recursive_search(problem):
    """Recursive depth first search. Kick-starts the recurse() function"""
    reached = set()
    return _recurse(Node(problem.initial), problem, reached)



if __name__ == '__main__':
    # Demo
    seeds = [3468, 5465, 4672, 2825, 5866, True]
    
    for seed in seeds:
        print(f"\n\nseed = {seed}:")
        graph = generate_graph(n=10, seed=seed)
        
        display_graph(graph)
        
        problem = SearchProblem(graph)
        
        # BFS
        solution = search(problem, "BFS")
        print("solution:", solution, end="\t\t")
        
        if solution:
            solution_path = problem.get_solution_path(solution)
            print("BFS solution path:", solution_path)
        
        
        # DFS
        solution = search(problem, "DFS")
        print("solution:", solution, end="\t\t")
        
        if solution:
            solution_path = problem.get_solution_path(solution)
            print("DFS solution path:", solution_path)
            
        
        # RECURSIVE SEARCH
        solution = recursive_search(problem)
        print("solution:", solution, end="\t\t")
        
        if solution:
            solution_path = problem.get_solution_path(solution)
            print("Recursive solution path:", solution_path)
