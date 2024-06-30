#!/usr/bin/env python3

#AIMA_page_116_eight_puzzle.py

"""
Solve an 8-puzzle with A* (with the Manhatten heuritic)

State is just a Python 1D tuple
"""


import random
from math import sqrt
import matplotlib.pyplot as plt

from enum import Enum, auto

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




# 1. Generate a random state
def generate(puzzle_name='eight', random_state=None):
    """."""
    if puzzle_name not in ('eight', 'fifteen'):
        raise ValueError("bad puzzle name")
    
    random.seed(random_state)
    state = list(range({'eight': 9, 'fifteen': 16}[puzzle_name]))
    random.shuffle(state)
    return tuple(state)
    


# 2. Validate a state whetehr it is valid and solvable
def validate(state):
    raise NotImplementedError("This function is not yet implemented")



# 3. Display a state
def display(state):
    """Display a state"""
    n = len(state)
    ax = plt.axes()
    d = [1/sqrt(n)*i for i in range(int(sqrt(n))+1)]
    ax.vlines(d, ymin=0.0, ymax=1.0, color='k')
    ax.hlines(d, xmin=0.0, xmax=1.0, color='k')
        
    d = [1/(int(sqrt(n)*2))*i for i in range(1, int(sqrt(n)*2), 2)]

    for ix, value in enumerate(state):
        if value==0:
            continue
        i,j = divmod(ix, int(sqrt(n)))
        ax.annotate(text=value, xy=(d[j], d[int(sqrt(n))-1-i]),
                    fontsize='xx-large', fontweight='bold',
                    horizontalalignment='center', verticalalignment='center')
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    return ax




# 4. Data structures for the problem reresentation

class PriorityQueue:
    """
    priority queue implemented with a binary heap.
    Accepts an evaluation function f to compare the nodes
    """
    def __init__(self, evaluation_function):
        self._nodes = []
        if not callable(evaluation_function):
            raise TypeError("The evaluation_function must be callable")
        self.f = evaluation_function

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
            c = r if (l and r) and (self.f(nodes[r]) < self.f(nodes[l])) else l
            
            # swap if necessary
            if c and self.f(nodes[c]) < self.f(nodes[p]):
                nodes[p], nodes[c] = nodes[c], nodes[p]
                p = c
            else: break
        return item



class Action(Enum):
    RIGHT = (0, 1)
    LEFT = (0, -1)
    TOP = (-1, 0)
    BOTTOM = (1, 0)

  
"""
@dataclass
class Node:
    state: Union[list, tuple]
    parent: Optional['Node'] = None
    action: Action = None
    path_cost: Optional[float] = 0.0
    heuristic: Optional[float] = float('inf')
    def __repr__(self): return "{}({})".format(self.__class__.__name__, self.state)
"""




class Node:
    def __init__(self, 
                    state: Union[list, tuple],
                    parent: Optional['Node'] = None,
                    action: Action = None,
                    path_cost: Optional[float] = 0.0,
                    heuristic: Optional[float] = float('inf')
                    ):
        self.state = tuple(state)  # to ensure imutability
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic = heuristic

    def __repr__(self): 
        return "{}({})".format(self.__class__.__name__, str(self.state)[1:-1])



# 5. Problem representation

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
        "returns the cost of performing an action from the current state that takes the agent to the next_state"
        return 1.0
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"



class SearchProblem(AbstractSearchProblem):
    """
    Concrete search problem representation and implementation.
    """
    def __init__(self, initial_state):
        """
        """
        self._initial_state = initial_state
        self.empty_tile = 0
        self.side_length = int(sqrt(len(self._initial_state)))
        
    @property
    def initial(self):
        return self._initial_state
    
    def is_goal(self, state) -> bool:
        return tuple(state) == tuple(range(len(self._initial_state)))
    
    def actions(self, state) -> set:
        i,j = divmod(state.index(self.empty_tile), self.side_length)
        
        candidates = {action: (i + action.value[0], j + action.value[1]) 
                      for action in Action}
        return {action for action, coords in candidates.items() 
                if min(coords) >= 0 and max(coords) < self.side_length}
        
    def result(self, state, action):
        zero_tile_index = state.index(self.empty_tile)
        i,j = divmod(state.index(self.empty_tile), self.side_length)
        i += action.value[0]
        j += action.value[1]
        assert min(i,j) >= 0 and max(i,j) < self.side_length, "bad action"
        swap_tile_index = i * self.side_length + j
        new_state = list(state)
        new_state[zero_tile_index], new_state[swap_tile_index] = new_state[swap_tile_index], new_state[zero_tile_index]
        return tuple(new_state)
    
    def heuristic(self, state):
        """the h2 heuristic as described in AIMA on page 116"""
        return 1.0
        
    def expand(self, node):
        state = node.state
        for action in self.actions(state):
            next_state = self.result(state, action)
            path_cost = node.path_cost + self.action_cost(state, action, next_state)
            heuristic = self.heuristic(next_state)
            yield Node(state=next_state, parent=node, action=action, 
                       path_cost=path_cost, heuristic=heuristic)
    
    def get_solution_path(self, node):
        assert self.is_goal(node.state), "node must be the GOAL"
        path = [node.state,]
        while node.state != self.initial:
            node = node.parent
            path.append(node.state)
        return path[::-1]
        




state = generate('fifteen')

node = Node(state)

display(state)



problem = SearchProblem(initial_state=state)


"""
actions = problem.actions(state)
print(actions)

for action in actions:
    new = problem.result(state, action)
    plt.figure()
    display(new)
"""


for node in problem.expand(node):
    print(node)
    plt.figure()
    display(node.state)




# 6. Search algorithm
...





#7. Demo test
if __name__ == '__main__':
    ...
    


