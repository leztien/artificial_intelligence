"""
convert a maze (represented as str or matrix) to an adjacency matrix
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def maze_to_matrix(str):

    l = str.splitlines()
    
    # Check whether all lines are equal in length
    if len(set(len(row) for row in l)) > 1:
        from warnings import warn
        warn("number of characters in each row is not equal", Warning)
    
    # Make an empty matrix
    m,n = len(l), max(set(len(row) for row in l))
    mx = np.empty(shape=(m,n), dtype=np.uint8)
    
    # Populate the matrix
    for i,row in enumerate(l):
        for j,c in enumerate(row):
            mx[i,j] = 1 if c=='#' else 0
    return mx


def get_neighbors(index: tuple, maze: np.ndarray):
    if maze[index]:
        raise IndexError("the index must not be an obstacle/wall")
    m,n = len(maze), len(maze[0])
    candidates = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
    valid_neighbors = []
    for e in candidates:
        r,c = ix = tuple(np.array(e) + index)
        if 0 <= r < m and 0 <= c < n and not maze[ix]:
            valid_neighbors.append(ix)
    return valid_neighbors


def maze_to_adjacency_matrix(maze, start: tuple):
    """render an adjacency matrix from a string/matrix representation of a maze (recursively)"""
    
    if isinstance(maze, str):
        maze = maze_to_matrix(maze)

    # Make an empty adjacency matrix (with exessive rows/columns)
    m,n = len(maze), len(maze[0])
    adjacency_matrix = np.zeros(shape=(m*n, m*n), dtype=np.uint8)
    
    k = -1  # number of found nodes
    d = dict()
    explored = set()
    
    def explore(ix):
        nonlocal maze, adjacency_matrix, k, d, explored
        
        # Base case
        if ix in explored:
            return
        
        # Register the node
        explored.add(ix)
        if ix not in d:
            k += 1
            d[ix] = k
            
        # Get neighbors
        neighbors = get_neighbors(ix, maze)
            
        # Recursive
        for e in neighbors:
            if e not in d:
                k += 1
                d[e] = k
                
            adjacency_matrix[d[ix], d[e]] = 1
            explore(e)
    
    # Recursively explore
    explore(start)

    # clip the adjacency matrix
    return adjacency_matrix[:k+1, :k+1]
            
###########################################################################################
  
    
maze = """\
#######
#     #
# ##  #
# #  ##
#     #
##### #"""

mx = maze_to_adjacency_matrix(maze, start=(1,1))
print(mx)

G = nx.from_numpy_matrix(mx)
nx.draw(G, with_labels=True)
