"""
DFS (no frills)
"""


from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



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
        mx = rs.rand(n,n)
        mx = np.triu(mx)
        mx += mx.T
        mx = (mx <= p).astype("uint8")
        mx[np.diag_indices(n)] = 0
        assert (mx == mx.T).all()
        
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
    nx.draw(G, with_labels=True)


def make_nodes_list(mx):
    nodes = [Node(node_id=i) for i in range(len(mx))]
    nodes[0].node_type = NodeType.INITIAL
    nodes[-1].node_type = NodeType.GOAL
    
    for row, node in zip(mx, nodes):
        node.neighbors = [nodes[ix] for ix,value in enumerate(row) if value]
    return nodes


############################################################


class NodeType(Enum):
    INITIAL = 0
    GOAL = 1
    NORMAL = 2


class Node:
    def __init__(self, neighbors=None, node_id=None, node_type=None):
        self.neighbors = neighbors or []
        self.node_id = node_id
        self.node_type = node_type or NodeType.NORMAL
        self.parent = None
    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.node_id)
    def __repr__(self):
        return self.__str__()        


def dfs(nodes):
    frontier = [nodes[0],]
    explored = []
    
    while frontier:
        # Remove a node from the frontier (and put it into the explored list)
        current_node = frontier.pop()
        explored.append(current_node)
        # If the removed node contains the goal state - return the solution
        if current_node.node_type is NodeType.GOAL:
            # Construct path
            path = [current_node,]
            while current_node.node_type != NodeType.INITIAL:
                current_node = current_node.parent
                path.append(current_node)
            return path[::-1]
        # Expand the current node by adding the (enexplored) neighbours to the frontier
        neighbors = [neighbor for neighbor in current_node.neighbors 
                     if neighbor not in explored and neighbor not in frontier]
        # Assign parent
        for node in neighbors:
            node.parent = current_node
        frontier.extend(neighbors)
    return None



# DEMO
if __name__ == '__main__':
    
    # generate a random adjacency matrix
    mx = make_adjacency_matrix(n_nodes=10, seed=True)
    print(mx)
    
    # display the graph
    display_graph(mx)
    
    # make a collection of nodes
    nodes = make_nodes_list(mx)
    
    solution = dfs(nodes)
    print("\nsolution path:", solution)
