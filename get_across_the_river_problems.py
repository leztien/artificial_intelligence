
"""
Solve the "Get across the river" problems, like
Missionaries and Cannibals,
Wolf, Goat and Cabbage
"""


from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Callable


##### Helper functions #####

def get_conditioned_tuples(bounds, min_sum=0, max_sum=float('inf')):
    """
    Generic helper function.
    Yield tuples that meet these conditions:
        - length == len(bounds)
        - each digit in the tuple is from 0 to bounds[index of digit]
        - the sum of the tuple is between min_sum and max_sum
    """
    def recurse(t):
        if len(t) == len(bounds):  # base case
            yield t
            return
        for digit in range(bounds[len(t)] + 1):
            yield from recurse(t + (digit,))
    
    for t in recurse(tuple()):
        if min_sum <= sum(t) <= max_sum:
            yield t
        if t[0] >= max_sum:
            break
        


##### Data structures (for the Problem representation and search algorithms) #####

class AbstractState(ABC):
    """Abstract class for a State"""
    def __init__(self, representation=None, *,
                       initial=None, goal=None,
                       actions=None):
        self.representation = representation  # atomic=id / factored / structural
        self.initial = initial  # bool
        self.goal = goal  # bool
        self.actions = actions
    
    def __hash__(self):
        """Necessary for __eq__. Rewrite as necessary"""
        return hash(self.representation)

    def __eq__(self, other):
        """Generic comparison. Rewrite as necessary"""
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.representation)

    def __str__(self):
        return f"{self.__class__.__name__}({repr(self.representation)})"
    

def state_class_factory(initial_state):
    """
    Factory function returning a Class.
    initial_state denotes the upper bounds
    """
    # Define a sub-class
    class State(AbstractState):
        _initial_state = tuple(initial_state)
        
        def __init__(self, representation: Tuple, **kwargs):
            if len(representation) != len(self._initial_state):
                raise ValueError("the provided argument must be "
                                 f"a sequence of length {len(self._initial_state)}")
            kwargs['goal'] = kwargs.get('goal') or set(representation) == {0}
            super().__init__(representation=tuple(representation), **kwargs)
            
            # Some assertions
            assert all(0<=x<=y for x,y in zip(self, self._initial_state))

        def __invert__(self):
            assert len(self._initial_state) == len(self.representation)
            return self.__class__([x - y for x,y in zip(self._initial_state, self.representation)])
        
        def __len__(self):
            return len(self.representation)
        
        def __iter__(self):
            yield from self.representation
        
        def __add__(self, seq):
            if isinstance(seq, self.__class__):
                raise TypeError(f"the object added cannot be a {self.__class__.__name__}")
            assert len(self) == len(seq), "the sequance added must be the same length as the state"
            assert seq[0] == (1 if self[0]==0 else -1)
            assert all(e>=0 for e in seq) or all(e<=0 for e in seq)
            return self.__class__([x + y for x,y in zip(self, seq)])
        
        def __getitem__(self, index):
            return self.representation[index]
        
        def get_text_art(self):
            d = {3: "BMC", 4: "BWGC", None: ''.join(chr(65+i) for i in range(len(self)))}
            chars = d.get(len(self), d[None])
            n = max(self._initial_state)
            s = '\n'.join((c*r).rjust(n) + " | " +(c*l).ljust(n) for c,r,l in zip(chars, self, ~self))
            return f"\n{s}\n"
    return State


class Node:
    """
    A Node is a wrapper around a State.
    """
    def __init__(self, state=None,
                       parent=None, children=None,
                       action=None, path_cost=None, heuristic=None,
                       id=None):
        self.state = state         # State
        self.parent = parent       # None or Node
        self.children = children   # None or []
        self.action = action       # tuple
        self.path_cost = path_cost # not used here
        self.heuristic = heuristic # not used here
        self.id = id    # ordinal number as int (for networx graph)

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id or self.state})"
    
    
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



##### Search algorithm #####

def bfs(problem):
    """
    BFS with that also checks weather a state is legal
    """
    node = Node(problem.initial)
    
    if problem.is_goal(node.state):
        return node
    
    frontier = Queue([node,])
    reached = {problem.initial}  # a set of states
    
    while frontier:
        node = frontier.pop()
        for child in problem.expand(node):
            state = child.state
            # If the state is not legal then skip this node
            if not problem.is_legal(state):
                continue
            if problem.is_goal(state):
                return child
            if state not in reached:
                reached.add(state)
                frontier.add(child)
    # return failure
    return None



##### Problem representation #####

class AbstractSearchProblem(ABC):
    """
    Abstract Search Problem class
    """
    def __init__(self):
        ...
    
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



class GetAcrossTheRiverProblem(AbstractSearchProblem):
    """
    Abstract class for "Get across the river" problems
    """
    def __init__(self, initial_state, boat_capacity):
        self._initial = initial_state
        try:
            self.min_boat_capacity, self.max_boat_capacity = boat_capacity
        except TypeError:
            self.min_boat_capacity, self.max_boat_capacity = boat_capacity, boat_capacity
    
    @property
    def initial(self):
        return self._initial
    
    def is_goal(self, state):
        return tuple(state.representation) == (0,) * len(state._initial_state)
    
    @abstractmethod
    def is_legal(self, state):
        """Implement concrete logic for a legal state"""
        return True
    
    def actions(self, state):
        """Check if this implementation can be inherited"""
        # from the west to east bank
        if state[0] == 1:
            actions = [(1,) + t for t in 
                       get_conditioned_tuples(state[1:], 
                                              self.min_boat_capacity, 
                                              self.max_boat_capacity)]
            actions = [tuple(-x for x in t) for t in actions]
        # from the east to west bank
        else:
            actions = [(1,) + t for t in 
                       get_conditioned_tuples((~state)[1:], 
                                              self.min_boat_capacity, 
                                              self.max_boat_capacity)]
        # Sort the actions to prefer the most effective
        #actions = sorted(actions, key=lambda t: abs(sum(t[1:])), reverse=bool(state[0]))
        # Some quick and dirty assertions
        assert all(min(state + action) >= 0 for action in actions)
        assert all(all(x<=y for x,y in zip(state + action, state._initial_state)) for action in actions)
        assert all(all(e<=0 for e in action) or all(e>=0 for e in action) for action in actions)
        assert len(actions) > 0
        assert all(a[0] == (1 if state[0]==0 else -1) for a in actions)
        return actions

    def result(self, state, action):
        """state is assumed to implement __add__"""
        return state + action
        
    def heuristic(self, state):
        """A very simple heuristic"""
        return sum(state)

    def expand(self, node):
        state = node.state
        for action in self.actions(state):
            next_state = self.result(state, action)
            path_cost = (node.path_cost or 0) + self.action_cost(state, action, next_state)
            heuristic = self.heuristic(next_state)  # path_cost and heuristic are not used here
            yield Node(state=next_state, parent=node, action=action, 
                       path_cost=path_cost, heuristic=heuristic)
    
    def solve(self):
        """
        Using BFS, as noted in Classic Computer Science Problems in Python on page 50
        """
        return bfs(self)
    
    def get_solution_path(self, node):
        assert node is not None and self.is_goal(node.state), "node must be the GOAL"
        path = [node.state.get_text_art(), self.decode_action(node.action)]
        while node.state != self.initial:
            node = node.parent
            path.extend([node.state.get_text_art(), self.decode_action(node.action)])
        return path[-2::-1]
    
    def decode_action(self, action: Tuple) -> str:
        """Convert a tuple denoting an action into a string, giving instructions"""
        if not action:
            return
        boat, *items = action
        items = [abs(item) for item in items]
        bank1, bank2 = ("west", "east")[::boat]
        bank = f" from the {bank1} bank to the {bank2} bank"
        items = ', '.join([f"""{n} {chr(66+i)}""" for i,n in enumerate(items)])
        return "Get " + items + bank



class MissionariesAndCannibals(GetAcrossTheRiverProblem):
    """
    one boat, 3 missionaries, 3 cannibals
    """
    def is_legal(self, state):
        for s in (state, ~state):
            if 0 < s[1] < s[2]:
                return False
        return True

    def decode_action(self, action: Tuple) -> str:
        if not action or len(action) != 3:
            return action
        b, m, c = action
        m, c = abs(m), abs(c)
        b1, b2 = ("west", "east")[::b]
        b = f"from the {b1} bank to the {b2} bank"
        d = {0: "", 1: "1 missionary", None: f"{m} missionaries"}
        m = d.get(m, d[None])
        d = {0: "", 1: "1 cannibal", None: f"{c} cannibals"}
        c = d.get(c, d[None])
        s = ("the boat", "", "and")[int(bool(m) + bool(c))]
        return ' '.join(filter(bool, ["Get", m, s, c , b])) + '.'

        
class WolfGoatCabbage(GetAcrossTheRiverProblem):
    """Wolf, Goat and Cabbage"""
    def is_legal(self, state):
        nono = [(0,1,1,0), (0,0,1,1), (0,1,1,1)]
        for s in (state, ~state):
            if tuple(s) in nono:
                return False
        return True

    def decode_action(self, action: Tuple) -> str:
        """Convert a tuple denoting an action into a string, giving instructions"""
        if not action:
            return
        boat, *items = action
        items = [abs(item) for item in items]
        bank1, bank2 = ("west", "east")[::boat]
        bank = f" from the {bank1} bank to the {bank2} bank"
        words = [[f"the {s}", ", "] for i,s in zip(items, ["wolf", "goat", "cabbage"]) if i]
        words = sum(words, [])[:-1]
        if len(words) > 1:
            words[-1:-2] = [" and "]
        return "Get " + ''.join(words) + bank

        

class Demo(GetAcrossTheRiverProblem):
    """For demonstartion purposes"""
    def is_legal(self, state):
        """Some random legal state checker"""
        return sorted(state[1:]) == list(state[1:]) \
               and sorted((~state)[1:]) == list((~state)[1:])
    
    
    


if __name__ == '__main__':

    ##### Missionaries and cannibals #####
    print("\nMissionaries and cannibals".upper())
    initial_state_representation = (1, 3, 3)  # 1 boat, 3 missionaries, 3 cannibals
    boat_capacity = (1, 2)  # min, max
    
    State = state_class_factory(initial_state_representation)
    initial_state = State(initial_state_representation)
    
    problem = MissionariesAndCannibals(initial_state, boat_capacity)
    solution = problem.solve()
    path = problem.get_solution_path(solution)
    
    for e in path:
        print(e)
    
    
    ##### Wolf, Goat, Cabbage #####
    print("\n\n\nWolf, Goat, Cabbage".upper())
    initial_state_representation = (1, 1,1,1)
    boat_capacity = (0, 1)
    
    State = state_class_factory(initial_state_representation)
    initial_state = State(initial_state_representation)
    
    problem = WolfGoatCabbage(initial_state, boat_capacity)
    solution = problem.solve()
    path = problem.get_solution_path(solution)
    
    for e in path:
        print(e)
    
    
    
    ##### A generic "Get across the river" problem #####
    print("\n\n\nGeneric demo".upper())
    print("(A = boat, B,C,D,E = four groups of players)")
    initial_state_representation = (1, 2,3,4,5)  
    boat_capacity = (2, 6)  # at least 2 ppl must steer the boat
    
    State = state_class_factory(initial_state_representation)
    initial_state = State(initial_state_representation)
    
    problem = Demo(initial_state, boat_capacity)  
    solution = problem.solve()
    
    path = problem.get_solution_path(solution)
    for e in path:
        print(e)
