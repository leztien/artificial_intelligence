# greedy_best_first_vs_a_star.py 
# best_first_vs_greedy_vs_a_star.py


"""
Greedy-Best-first search vs A*
"""

from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple, Union, Optional, Any
from dataclasses import dataclass
from math import atan, degrees
from math import sqrt, floor, ceil
from itertools import product
from functools import reduce
from operator import or_
import numpy as np
import matplotlib.pyplot as plt



##### Random problem generation and vizualization functions #####

def generate_graph(n=10, random_state=None):
    """
    Generates a random eucledian graph with n verteces and 
    approximately edge_density * (n*(n-1)/2) edges.
    Returns a dictionary in this form:
    graph = {'vertices': {"A": (x, y), "B": (x, y)}, 'edges': {("A","B"): distance}}
    """
    
    def generate_grid_points(n, height=10, width=15, random_state=None):
        """make a coordinates matrix of random points based on a grid"""
        rs = np.random.RandomState(random_state) if type(random_state) in (int, type(None)) else random_state
        side_factor = width / height
        side = sqrt(n / side_factor)
        h, w = ceil(side), ceil(side*side_factor)
        
        xx = [width / (w-1) * i for i in range(w)]
        yy = [height / (h-1) * i for i in range(h)]

        coordinates_matrix = np.array(list(product(xx, yy))[:n])
        coordinates_matrix += rs.normal(0, scale=width/20, size=(n,2))
        coordinates_matrix -= coordinates_matrix.min(axis=0) - 1
        coordinates_matrix = coordinates_matrix // 0.5 * 0.5
        assert len(coordinates_matrix) == n
        return coordinates_matrix
    
    # random state
    random_state = int(random_state or np.random.randint(1, 9999))
    print("random state:", random_state)
    rs = np.random.RandomState(random_state)
    
    assert n <= 26, "n must not be greater than the number of letters in the latin alphabet"
    
    # some preliminary constants
    edge_density = 0.1
    span = 10

    # this will be returned    
    graph = {'vertices': dict(), 'edges': dict()}
    
    # generate coordinates of the n points
    if rs.rand() < 0.5:
        coordinates_matrix = (rs.rand(n, 2) * span) // 0.5 * 0.5
    else:
        coordinates_matrix = generate_grid_points(n, random_state=rs)
    
    graph['vertices'] = {chr(65 + i): (x,y) for i, (x,y) in enumerate(coordinates_matrix)}
    
    # generate edges
    probabilities_matrix = rs.rand(n, n)
    probabilities_matrix[np.diag_indices(n)] = 0
    
    while edge_density < 1.0:
        adjacency_matrix = np.triu(probabilities_matrix) >= (1 - edge_density)
        edge_density += 0.001
        
        if (adjacency_matrix + adjacency_matrix.T).sum(axis=0).min() > 0:
            break
    
    for i,row in enumerate(adjacency_matrix):
        for j in np.nonzero(row)[0]:
            distance = ((coordinates_matrix[i] - coordinates_matrix[j]) ** 2).sum() ** 0.5
            graph['edges'][(chr(65 + i), chr(65 + j))] = round(distance)
    return graph



def draw_graph(graph, path=None):
    """Draws a geometric graph from the given graph dictionary"""
    
    # Some inner helper functions
    def plot_point(point, **kwargs):
        params = dict(marker = 'o', color='blue', zorder=0)
        params.update(kwargs)
        plt.plot(*point, **params)
        
    def annotate_point(point, text, **kwargs):
        params = dict(
            xytext=(-15, -5), textcoords='offset points',
            fontsize='x-large', fontweight='bold', zorder=1
            )
        params.update(kwargs)
        plt.annotate(text=text, xy=point, **params)
    
    def plot_line(point1, point2, **kwargs):
        params = dict(linestyle = '-', color='black', zorder=-1)
        params.update(kwargs)
        plt.plot(*zip(point1, point2), **params)
        
    def annotate_line(point1, point2, text, **kwargs):
        params = dict(xytext=(0, 10), 
                     horizontalalignment='center',
                     verticalalignment='center',
                     textcoords='offset points', fontsize='medium', color='black')
        params.update(kwargs)
        
        x, y = (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2
        slope = (point2[1] - point1[1]) / ((point2[0] - point1[0]) or 1E-9)  #to avoid devision by zero
        plt.annotate(text=text, xy=(x, y), rotation=degrees(atan(slope)), **params)
    
    # draw the grapg parts
    plt.figure()
    
    # plot and annotate each vertex i.e. point
    for label, point in graph['vertices'].items():
        color='green' if label=='A' else 'red' if label==chr(64+len(graph['vertices'])) else 'blue'
        plot_point(point, color=color)
        annotate_point(point, text=label, color='black')
    
    # draw and annotate each line i.e. edge
    for (point1, point2), value in graph['edges'].items():
        point1 = graph['vertices'][point1]
        point2 = graph['vertices'][point2]
        plot_line(point1, point2)
        if not path: annotate_line(point1, point2, text=value)
        
    # darw the path
    if path:
        points = [graph['vertices'][k] for k in path]
        for i in range(len(points)-1):
            plot_line(points[i], points[i+1], color='red')
            annotate_line(points[i], points[i+1], text=f"{i+1}", weight='bold')
    
    plt.axis('equal')
    return plt.gca()



##### Helper data structure(s) #####

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


##### Problem data structures #####

"""
class State(NamedTuple):
    "immutable"
    id: Union[int, str]  # e.g. State(1), State('INITIAL')
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.id)
    def __str__(self): return f"{self.id}"
    def __hash__(self): return hash(self.id)
"""

@dataclass
class Node:
    state: Union[int, str, Any]
    parent: Optional['Node'] = None
    action: Optional[Any] = None
    path_cost: Optional[float] = float('inf')
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.state)
    def __lt__(self, other): return self.path_cost < other.path_cost

"""
class Action:
    "an action can be any appropriate object, str, int, float, custom structured object, etc"
    def __init__(self, go_to_state): self.go_to_state = go_to_state
    def __str__(self): return "{}(go to {})".format(self.__class__.__name__, self.go_to_state)
    def __repr__(self): return self.__str__()
"""   
    
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
        "returns a new state"
    
    @abstractmethod
    def action_cost(state, action, next_state) -> float:
        "returns the cost of performing an action from the current state that takes the agent to the next_state"


class SearchProblem(AbstractSearchProblem):
    """
    Concrete search problem representation and implementation.
    """
    def __init__(self, problem_representation):
        """
        Expects a euclidean graph as problem representation, in this format:
        graph = {'vertices': {"A": (x, y), "B": (x, y)}, 'edges': {("A","B"): distance}}
        """
        self.graph = problem_representation
        self.graph['edges'] = {frozenset(k):v for k,v in self.graph['edges'].items()}

    @property
    def initial(self):
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



##### Solver #####




class Solver:
    
    def best_first(problem, f):
        """
        best first search with a priority queue (based on an evaluation function f)
        AIMA page 91
        """
        node = Node(state=problem.initial)
        frontier = PriorityQueue(f) # f = evaluation function 
        frontier.add(node)
        reached = {problem.initial: node}
        
        while not frontier.is_empty():
            node = frontier.pop()
            if problem.is_goal(node.state):
                return node  # from which the solution path can be constructed
            
            for child_node in problem.expand(node):
                state = child_node.state
                if state not in reached or child_node.path_cost < reached[state].path_cost:
                    reached[state] = child_node
                    frontier.add(child_node)
        return 'failure'


    def greedy(self, problem):
        ...
        
    def a_star(self, problem):
        ...
    
    
    def expand(self, node):
        """
        in AIMA, expand() is a stand-alone function
        """
        state = node.state
        for action in self.actions(state):
            next_state = self.result(state, action)
            cost = node.path_cost + self.action_cost(state, action, next_state)
            yield Node(state=next_state, parent=node, action=action, path_cost=cost)
        

    def evaluation_function(node):
        """
        evaluation function f
        to compare nodes in the priority queue.
        In AIMA the evaluation function f is a stand-alone function.
        """
        return node.path_cost






if __name__ == '__main__':
    
    graph = generate_graph(10, random_state=None)
    print(graph)
    
    ax = draw_graph(graph)
    
    problem = SearchProblem(graph)
    
    solver = Solver()
    
    solution = solver.greedy(problem)
    
    solution_path = None
    draw_graph(graph, path=solution_path)
    


