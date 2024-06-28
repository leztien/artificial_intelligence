

# AIMA_page_095_bfs_vs_dfs.py

"""
Breadth Firs search
iterative Depth First search
recursive Depth First search
"""



from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple, Union, Optional, Any
from dataclasses import dataclass
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
    return G
    
    
    
#### Data Structures to fomralize the problem #####

class AbstractQueue(ABC):
    """Abstract Queue for FIFO and LIFO queues"""
    def __init__(self, items=None): self._items = list(items or [])
    def __repr__(self): return f"{self.__class__.__name__}({str(self._items)[1:-1]})"
    def __getitem__(self, index): return self._items[index]
    def __delitem__(self, index): del self._items[index]
    def __len__(self): return len(self._items)
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
    path_cost: Optional[float] = 0.0
    heuristic: Optional[float] = float('inf')
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
    
    @abstractmethod
    def action_cost(state, action, next_state) -> float:
        "returns the cost of performing an action from the current state that takes the agent to the next_state"


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
        #self.graph['edges'] = {frozenset(k):v for k,v in self.graph['edges'].items()}

    @property
    def initial(self):
        """Returns the INITIAL state"""
        return min(self.graph['vertices'].keys())
    
    def is_goal(self, state) -> bool:
        return state == max(self.graph['vertices'].keys())
    
    def actions(self, state) -> set:
        return reduce(or_, {edge for edge in self.graph['edges'] if state in edge}, frozenset()).difference({state})
        
    def result(self, state, action):
        """transition model"""
        next_state = action  # here
        edge = frozenset({state, action})
        return next_state if edge in self.graph['edges'] else None
    
    def action_cost(self, state, action, next_state=None) -> float:
        assert next_state is None or next_state == action, "action must be == next_state in this problem type implementation"
        edge = frozenset({state, action})
        return self.graph['edges'][edge]

    def heuristic(self, state):
        """get the heuristic value from the given node to the GOAL"""
        state_coords = np.array(self.graph['vertices'][state])
        goal_coords = np.array(self.graph['vertices'][max(self.graph['vertices'])])
        return ((goal_coords - state_coords) ** 2).sum() ** 0.5
        
    def expand(self, node):
        """
        note: in AIMA, expand() is a stand-alone function
        """
        state = node.state
        for action in self.actions(state):
            next_state = self.result(state, action)
            path_cost = node.path_cost + self.action_cost(state, action, next_state)
            heuristic = self.heuristic(next_state)
            yield Node(state=next_state, parent=node, action=action, 
                       path_cost=path_cost, heuristic=heuristic)
    
    def get_solution_path(self, node, return_cost=False):
        assert self.is_goal(node.state), "node must be the GOAL"
        path = [node.state,]
        while node.state != self.initial:
            node = node.parent
            path.append(node.state)
        path = path[::-1]
        
        if return_cost:
            edges = [frozenset({path[i], path[i+1]}) for i in range(len(path)-1)]
            cost = sum(self.graph['edges'][edge] for edge in edges)
        
        return (path, cost) if return_cost else path
    

 





##### The most important part - the search functions themselves #####

def search(problem, search_type="bfs"):
    ...



def recursive_dfs(problem):
    ...






if __name__ == '__main__':
    # Demo
    graph = generate_graph(n=10, seed=True)
    
    display_graph(graph)
    
    
    
    




