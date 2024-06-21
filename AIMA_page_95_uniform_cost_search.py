#!/usr/bin/env python3

"""
Uniform-cost search aka Dijkstra's algorithm
AIMA page 95
"""


from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple, Union, Optional, Any
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def make_adjacency_matrix(n, density=None, seed=None):
    """makes a random adjacency matrix representing a graph"""
    p = density or np.log(n) / 10 / n
    
    # Random seed
    if seed is True:
        seed = np.random.randint(1,9999)
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
    
    # add edge weights
    n_edges = np.triu(mx).sum()
    w = np.triu(rs.randint(1, n_edges, size=(n, n)))
    w += w.T
    mx = mx * w
    assert (mx == mx.T).all()
    return mx


def display_graph(mx, relable=False):
    """converts an adjacency matrix with edge weights into a graph"""
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
    
    # relable the nodes
    if relable:
        mapping = {i:chr(65+i) for i in range(len(G))}
        G = nx.relabel_nodes(G, mapping)
    
    # positioning
    pos = nx.spring_layout(G, seed=None)  
    
    # nodes
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=300)
    
    # edges
    nx.draw_networkx_edges(G, pos, width=3)

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

###############################################################################


class PriorityQueue:
    """
    priority queue implemented with a binary heap.
    Accepts an evaluation function f to compare the nodes
    """
    def __init__(self, f=None):  # f = evaluation function to compare items: f(item)
        self._nodes = []
        self.f = f if callable(f) else lambda item: item

    def __len__(self):
        return len(self._nodes)
    
    def is_empty(self):
        return len(self._nodes) == 0

    def add(self, item):
        # alias for convenience
        nodes = self._nodes
        
        # append to the end
        nodes.append(item)
        
        # bubble up
        c = len(nodes) - 1
        while c:
            p = (c - (1 if c%2 else 2)) // 2
            if self.f(nodes[c]) < self.f(nodes[p]):
                nodes[c], nodes[p] = nodes[p], nodes[c]
                c = p
            else: break
    
    def top(self):
        """returns the top element but does not remove it, unlike pop()"""
        return self._nodes[0]
    
    def pop(self):
        # alias for convenience
        nodes = self._nodes
        
        # swap
        nodes[0], nodes[-1] = nodes[-1], nodes[0]
        
        # pop
        item = nodes.pop()
        
        # bubble down
        p = 0
        while True:
            # determine the indeces
            l = (2*p + 1) if (2*p + 1) < len(nodes) else None
            r = (2*p + 2) if (2*p + 2) < len(nodes) else None
            c = r if (l and r) and (nodes[r] < nodes[l]) else l
            
            # swap if necessary
            if c and self.f(nodes[c]) < self.f(nodes[p]):
                nodes[p], nodes[c] = nodes[c], nodes[p]
                p = c
            else: break
        return item
        

class State(NamedTuple):
    """immutable"""
    id: Union[int, str]  # e.g. State(1), State('INITIAL')
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.id)
    def __str__(self): return f"{self.id}"
    def __hash__(self): return hash(self.id)


@dataclass
class Node:
    """mutable"""
    state: State
    parent: Optional['Node'] = None
    action: Optional[Any] = None
    path_cost: Optional[float] = 0.0
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.state)
    def __lt__(self, other): return self.path_cost < other.path_cost


class Action:
    """an action can be any appropriate object, str, int, float, custom structured object, etc"""
    def __init__(self, go_to_state): self.go_to_state = go_to_state
    def __str__(self): return "{}(go to {})".format(self.__class__.__name__, self.go_to_state)
    def __repr__(self): return self.__str__()
    


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
        "based on the transition model"
    
    @abstractmethod
    def action_cost(state, action, next_state) -> float:
        "returns the cost of performing an action from the current state that takes the agent to the next_state"


class SearchProblem(AbstractSearchProblem):
    """
    Concrete search problem representation and implementation.
    This implementation of search Problem works with an adjacency matrix.
    __init__ creates a static state-space
    """
    def __init__(self, adjacency_matrix=None):
        # the first row represents the INITIAL, the last the GOAL state
        self.adjacency_matrix = adjacency_matrix
        self.statespace = tuple(State(i) for i,_ in enumerate(self.adjacency_matrix))
    
    @property
    def initial(self):
        return self.statespace[0]
    
    def is_goal(self, state) -> bool:
        # the last row in the adjecency matrix is assumed to represent the goal state
        return state.id == len(self.adjacency_matrix) - 1
    
    def actions(self, state):
        return [Action(self.statespace[i]) for i in np.nonzero(self.adjacency_matrix[state.id])[0]]
    
    def result(self, state, action):
        i,j = state.id, action.go_to_state.id
        if self.adjacency_matrix[i,j]:
            return self.statespace[j]
        return None
    
    def action_cost(self, state, action, next_state) -> float:
        assert action.go_to_state is next_state, "must be able to get to the new state with the action provided"
        i,j = state.id, next_state.id
        return self.adjacency_matrix[i,j]
    
    def expand(self, node):
        """
        The expand() function is a stand-alone function in AIMA implementation.
        I have decided to make it a method of a SearchProblem() instance,
        becuase it is the problem who must know how to expand one of its nodes
        bzw, states
        """
        state = node.state
        for action in problem.actions(state):
            next_state = problem.result(state, action)
            cost = node.path_cost + problem.action_cost(state, action, next_state)
            yield Node(state=next_state, parent=node, action=action, path_cost=cost)
        

def path_cost(node):
    """evaluation function for a node to compare nodes in the priority queue"""
    return node.path_cost


def uniform_cost_search(problem, f=path_cost):
    """
    uniform-cost search 
    AIMA page 95
    """
    node = Node(state=problem.initial)
    frontier = PriorityQueue(f) # f = evaluation function 
    frontier.add(node)
    reached = {problem.initial: node}
    
    while not frontier.is_empty():
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node  # from which the solution path can be constructed
        
        for child in problem.expand(node):  # child = child node
            state = child.state
            if state not in reached or child.path_cost < reached[state].path_cost:
                reached[state] = child
                frontier.add(child)
    return 'failure'




if __name__ == '__main__':
    # DEMO

    seeds = [7763, 6907, 2057, 685, 8169, 6577, 3577, 3641, 9983, True]
    
    for seed in seeds:
        mx = make_adjacency_matrix(10, seed=seed)
        
        display_graph(mx)
        
        problem = SearchProblem(mx)
        
        output = uniform_cost_search(problem, f=path_cost)
        print("solution:", output)
        
        # construct the solution path
        if output != 'failure':
            node = output
            solution_path = [node,]
            
            while not node.state is problem.initial:
                node = node.parent
                solution_path.append(node)
        
            solution_path = solution_path[::-1]
            print("solution path:", solution_path, end="\n\n")
