
"""
best first search
with weighted edges (path costs)

AIMA page 91
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass



def make_priority_queue():
    """
    Returns a priority queue instance
    """
    
    def load_module_from_github(url):
        from urllib.request import urlopen
        from tempfile import NamedTemporaryFile
        from os.path import split, dirname
        from sys import path
        
        obj = urlopen(url)
        assert obj.getcode()==200,"unable to open"
        
        with NamedTemporaryFile(mode='w+b', suffix='.py') as f:
            f.write(obj.read())
            f.seek(0)
        
            path.append(dirname(f.name))
            module = __import__(split(f.name)[-1][:-3])
        del obj
        return module

    try:
        url = r"https://raw.githubusercontent.com/leztien/computer_science/main/priority_queue.py"
        return load_module_from_github(url).Heap()
    
    except:
        from queue import PriorityQueue
        from warnings import warn
        
        # message
        warn("unable to load from github, falling back on queue.PriorityQueue", Warning)
        
        # monkey patching
        PriorityQueue.add = PriorityQueue.put
        PriorityQueue.pop = PriorityQueue.get
        PriorityQueue.is_empty = PriorityQueue.empty
        
        # instantiate and return
        return PriorityQueue()



def make_adjacency_matrix(n_nodes, density=None, seed=None):
    """makes a random adjacency matrix representing a graph"""
    n = n_nodes
    p = density or np.log(n_nodes) / 10 / n_nodes
    
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


def display_graph(mx):
    """converts an adjacency matrix with edge weights into a graph"""
    try:
        G = nx.from_numpy_matrix(mx)
    except AttributeError:
        G = nx.from_numpy_array(mx)

    # add coloring to the nodes
    color_map = []
    for node in G:
        if node == 0:
            color_map.append('blue')
        elif node == len(mx) - 1:
            color_map.append('red')
        else: 
            color_map.append('cyan')      
    
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
    
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



def make_statespace(mx):
    states = statespace = [State(i) for i in range(len(mx))]
    
    for s in states:
        s.type = None
    
    states[0].type = 'INITIAL'
    states[-1].type = 'GOAL'
    
    # neighbors
    for row, state in zip(mx, states):
        state.actions = [states[ix] for ix,value in enumerate(row) if value]
        state.costs = {states[ix]: value for ix,value in enumerate(row) if value}
    return states


############################################################


@dataclass
class State:
    id: int or str
    def __str__(self): return f"{self.id}"
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.id)
    def __hash__(self): return hash(self.id)
    
    
class Node:
    """
    Wrapper for a State
    """
    def __init__(self, state, parent=None, action=None, cost=None):
        self.state:  State = state
        self.parent: Node = parent
        self.action: object = action
        self.cost:   float = float(cost or 0.0)
        
    def __lt__(self, other): return self.cost < other.cost
    def __str__(self): return "{}({})".format(self.__class__.__name__, self.state)
    def __repr__(self): return self.__str__()



def is_goal(state):
    """concrete implementation"""
    if type(state) is not State:
        raise TypeError("the input argument must be an instance of State")
    return hasattr(state, 'type') and state.type == 'GOAL'
    

def actions(state):
    return state.actions # possible actions in this state


def result(state, action):
    """concrete implementation"""
    new_state = action if action in state.actions else None
    return new_state


def action_cost(state, action, new_state=None):
    """concrete implementation"""
    return state.costs[action]


def expand(statespace, node):
    state = node.state
    
    for action in actions(state):
        new_state = result(state, action)
        cost = node.cost + action_cost(state, action, new_state)
        yield Node(state=new_state, parent=node, action=action, cost=cost)
    

def best_first_search(statespace):
    """
    ...
    state-space corresponds to "problem in AIMA"
    """
    
    initial_state = statespace[0]
    
    node = Node(initial_state)  # the INITIAL
    frontier = make_priority_queue()
    frontier.add(node)
    reached = {initial_state: node}
    
    while not frontier.is_empty():
        node = frontier.pop()
        if is_goal(node.state):
            return node
        
        for child_node in expand(statespace, node):
            state = child_node.state
            if state not in reached or child_node.cost < reached[state].cost:
                reached[state] = child_node
                frontier.add(child_node)
    return 'failure'


def get_solution_path(node):
    if not (type(node) is Node and is_goal(node.state)):
        raise TypeError("The input argument must be a goal node")
    
    path = [node,]
    while not node.state.type == 'INITIAL':
        node = node.parent
        path.append(node)
    
    return path[::-1]




# DEMO
if __name__ == '__main__':
    
    seeds = [6907, 2057, 685, 8169, 6577, 3577, 3641, True]
    print("\n\nDemonstrating some interesting graphs:")
    
    for seed in seeds:
    
        # generate a random adjacency matrix
        mx = make_adjacency_matrix(10, seed=seed)
        
        # display the graph
        display_graph(mx)
        
        # make a statespace
        statespace = make_statespace(mx)
        
        # best first search
        solution = best_first_search(statespace)
        print("solution:", solution)
        
        # solution path
        path = get_solution_path(solution)
        print("solution path:", path, "\n\n")
    

