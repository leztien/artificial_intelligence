#!/usr/bin/env python3

"""
Solve an 8-puzzle with A* (with the Manhatten heuritic)

State is just a Python 1D tuple
"""

from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import Union, Optional
import random
from math import sqrt
import matplotlib.pyplot as plt



# 1. Generate a random state
def generate(puzzle_name='eight', random_state=None):
    """Gerenrate a random 8-puzzle state"""
    if puzzle_name not in ('eight', 'fifteen'):
        raise ValueError("bad puzzle name")
    
    # construct the INITIAL state
    state = list(range({'eight': 9, 'fifteen': 16}[puzzle_name]))
    
    # seed a random state for shuffling
    random.seed(random_state)
    
    # shuffle and check for solvability
    while not random.shuffle(state):  # in effect an infinite loop
        if validate(state):
            return tuple(state)
    


# 2. Validate a state whetehr it is valid and solvable
def validate(state):
    """
    Validate a state for solvability. Copied the code from ChatGPT:
    https://chatgpt.com/share/9488c652-821e-45b5-9186-2e9e819c1882
    """
    
    state = list(state)
    state.remove(0)
    
    # Count the number of inversions
    inversions = 0
    for i in range(0, len(state)-1):
        for j in range(i + 1, len(state)):
            if state[i] > state[j]:
                inversions += 1

    # Check if the number of inversions is even
    return inversions % 2 == 0




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
        self._initial = initial_state
        self.side_length = int(sqrt(len(self._initial)))
        self.empty_tile = 0
        
    @property
    def initial(self):
        return self._initial
    
    def is_goal(self, state) -> bool:
        return tuple(state) == tuple(range(len(self.initial)))
    
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
    
    @staticmethod
    def _manhattan_distance(a, b):
        """a and b are expected to be pairs of (x,y) coordinates"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def heuristic(self, state):
        """the h2 heuristic as described in AIMA on page 116"""        
        return sum(self._manhattan_distance(divmod(ix, self.side_length), divmod(value, self.side_length)) 
                   for ix, value in enumerate(state) if value != self.empty_tile)

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
        


# 6. Search algorithm

def best_first(problem, f):
    """
    Generic search algorithm.
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
        
        for child in problem.expand(node):  # child = child node
            state = child.state
            if state not in reached or child.path_cost < reached[state].path_cost:
                reached[state] = child
                frontier.add(child)
    
    # return failure
    return reached  # for introspection purposes
    
    

def g(node):
    """path cost"""
    return node.path_cost

def h(node):
    """heuristic h2 as defined in AIMA on page 116"""
    return node.heuristic


def uniform_cost(problem):
    """aka Dijkstra's algorithm"""
    return best_first(problem, f=g)
    
    
def greedy(problem):
    """uses the heuristic"""
    return best_first(problem, f=h)
    
    
def a_star(problem):
    """Used for the eight puzzle"""
    f = lambda node: g(node) + h(node)
    return best_first(problem, f=f)





#7. Demo test
if __name__ == '__main__':

    initial_state = generate(puzzle_name='eight', random_state=None)
    display(initial_state).set_title("INITIAL state"); plt.show()
    
    problem = SearchProblem(initial_state=initial_state)
    
    # Use A* (with the "h2" heuristic)
    solution = a_star(problem)
    
    if type(solution) is Node:
        print("GOAL state:", solution.state)
        solution_path = problem.get_solution_path(solution)
        
        for move, state in enumerate(solution_path):
            if move == 0: continue
            plt.figure()
            ax = display(state)
            ax.set_title(f"move: {move}")
    else:
        print(f"explored {len(solution)} states and didn't find the GOAL state")

