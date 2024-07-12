
"""
Alternating Recursion Tree-Search algorithm for a state-space with LOOPs and GOALs
aka
And-Or-Search

Important: notice that I have changed the logic in the and_search function, which seems to work correctly!
Compare the original pseudocode for the AND-SEARCH function in AIMA on page 143
"""


import random
import itertools
from types import SimpleNamespace

from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

try:
    from networkx.drawing.nx_agraph import graphviz_layout  # new
except ImportError:
    from networkx.drawing.nx_pydot import graphviz_layout   # old
    


##### HELPER FUNCTIONS #####

class State:
    def __init__(self, representation=None,
                       initial=None, goal=None,
                       actions=None):
        self.representation = representation  # atomic=id / factored / structural
        self.initial = initial  # bool
        self.goal = goal  # bool
        self.actions = actions

    def __hash__(self):
        return hash(self.representation)

    def __eq__(self, other):  # just very simple (primitive) eq
        return self.representation == other.representation

    def __repr__(self):
        return str(self.representation)

    def __str__(self):
        return f"{self.__class__.__name__}({repr(self.representation)})"


class Node:
    """
    A Node is a wrapper around a State.
    Here Node is used only in the and-or-graph drawing function
    """
    colors = SimpleNamespace(
                INITIAL = 'orange',
                GOAL = 'green',
                OR_NODE = 'orange',
                AND_NODE = 'lightblue',
                LOOP = 'red',
                TRUNCATED = 'lightgrey')
    
    def __init__(self, state=None,
                       parent=None, children=None,
                       action=None, path_cost=None, heuristic=None,
                       id=None, color=None, label=None):
        self.state = state         # State
        self.parent = parent       # None or Node
        self.children = children   # None or []
        self.action = action       # not used here
        self.path_cost = path_cost # not used here
        self.heuristic = heuristic # not used here
        self.id = id    # ordinal number as int
        self.color = color
        self.label = label

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"


def make_state_space(random_state=None, *,
                     n_states=10, n_goal_states=2,
                     n_actions=4, n_belief_states=3):
    """
    Creates a random state space for an AND-OR problem
    """
    # Random state for replication
    random_state = random_state or random.randint(1, 9999)
    print("random_state =", random_state)
    random.seed(random_state)
    assert n_states > n_goal_states

    states = [
        State(representation=i,
              initial= i==0,
              goal= i >= n_states - n_goal_states)
        for i in range(n_states)
        ]

    actions = [chr(65+i) for i in range(n_actions)]

    for state in states:
        state.actions = {action: set(random.sample(states, random.randint(1, n_belief_states)))
                         for action in random.sample(actions, random.randint(1, len(actions)))}
    return states



def print_states(states):
    """
    Prints the states and actions (for vizual analysis)
    """
    print("\033[31m" + "\nSTATE SPACE:\nstate\t\tactions" + "\033[0m")
    for state in states:
        d = state.actions
        s = ", ".join([c + ': {' + f"{sorted(state.representation for state in d[c])}"[1:-1] + '}' for c in sorted(d.keys())])
        print(state, "\t", s)
    print()




        
def draw_and_or_graph(states):
    """
    Draws an and-or graph
    """
    # Trying to get the n_levels not to cluter the graph
    n = len(sum([[len(e) for e in state.actions.values()] for state in states], []))
    max_levels = 3 if n < 23 else 2
    
    # Helper functions for alternating recursion
    def or_nodes_traversal(state, node, path, nodes, edges, counter):
        # Base cases
        if state.goal == True:
            node.color = Node.colors.GOAL
            return
        if is_cycle(state, path):
            node.color = Node.colors.LOOP
            return
        
        if len(path) >= max_levels:
            node.color = Node.colors.TRUNCATED
            node.label = chr(0x2026)   # "..."
            return
    
        # For-loop with recursions
        for action in sorted(state.actions):
            belief_state = state.actions[action]
            child = Node(id=next(counter), state=belief_state, color=Node.colors.AND_NODE)
            nodes.append(child); edges[(node.id, child.id)] = action
            and_nodes_traversal(belief_state, child, [state] + path, nodes, edges, counter)
    
    def and_nodes_traversal(belief_state, node, path, nodes, edges, counter):
        for state in belief_state:
            child = Node(id=next(counter), state=state, color=Node.colors.OR_NODE, label=str(state.representation))
            nodes.append(child); edges[(node.id, child.id)] = ''
            or_nodes_traversal(state, child, path, nodes, edges, counter)
    
    # Figure, Axe
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Instantiate the necessary objects
    counter = itertools.count()
    initial = states[0]  # INITIAL state
    root = Node(id=next(counter), state=initial, 
                color=Node.colors.INITIAL, label=str(initial.representation))
    nodes = [root,]
    edges = dict()
    path = list()
    
    # Kick start the recursion
    or_nodes_traversal(state=initial, node=root, path=path, 
                       nodes=nodes, edges=edges, counter=counter)
    
    # Make nx tree
    T = nx.from_edgelist(edges)
    pos = graphviz_layout(T, prog="dot")
    
    # Get colors for the nodes
    color_map = [node.color for node in sorted(nodes)]
    assert len(T) == len(nodes) == len(color_map)
    
    # Get the node labels
    labels = {i: str(node.label or '') for i,node in zip(T, nodes)}
    
    # Draw nodes, edges, node labels
    nx.draw_networkx_nodes(T, pos, node_color=color_map, node_size=250)
    nx.draw_networkx_edges(T, pos, width=1)
    nx.draw_networkx_labels(T, pos, labels=labels, font_size=12, font_family="sans-serif")
    nx.draw_networkx_edge_labels(T, pos, edge_labels=edges, rotate=False, font_color='orange', font_weight='bold', font_size=8)
    
    # Make a legend
    legend_handles = [
        Patch(color=Node.colors.OR_NODE, label='OR-node'),
        Patch(color=Node.colors.AND_NODE, label='AND-node'),
        Patch(color=Node.colors.GOAL, label='GOAL'),
        Patch(color=Node.colors.LOOP, label='LOOP'),
        Patch(color=Node.colors.TRUNCATED, label='truncated'),
    ]
    
    ax.legend(handles=legend_handles, title="Node colors:", loc='upper left')
    
    # Set the title and show the graph
    ax.set_title("And-Or Graph")
    plt.show()
    


##### THE MEAT OF THIS MODULE #####

def and_or_search(states):  # recursion kick-starter
    return or_search(states[0], path=list())


def or_search(state, path)-> list or None:
    GOAL = chr(0x1F44D)   # in AIMA pseudocode, GOAL = []
    
    # Base case
    if state.goal == True:
        return GOAL
    if is_cycle(state, path):
        return None

    # For-loop with recursions for each action
    for action in state.actions:
        belief_state = state.actions[action]
        
        # Recurse (each action i.e. each belief_state)
        plan = and_search(belief_state, [state] + path)
        
        # If any path from this or-node leads to a goal state then...
        if plan:
            # this if-block is to pretify the output
            if len(plan.keys()) == 1:
                return [action] + list(tuple(plan.values())[0])
            return [action] + [plan]

    # Return failure
    return None


def and_search(belief_state, path) -> dict or None:
    # Instantiate a dict representing a subplan
    d = dict()

    # For-loop with recursions for each state in the given belief state
    for state in belief_state:
        plan = or_search(state, path)
        
        # ...?
        if plan == None:
            return None
        d[state] = plan

    # Return the sub-plan (as dict)
    return d


def is_cycle(state, path):
    return state in path




# Demo
if __name__ == '__main__':
    
    # Random states for some interesting and-or graphs
    seeds = [
        9654, 9509, 1960, 6213, 2548, 1057, 3829, 7099, 8754, 28, 4823, 7793, 
        1564, 2959, 221, 6471, 2744, 
        ]
    
    # Substitute None with a number from the list above to see an interesting case
    seed = None
    states = make_state_space(seed)

    print_states(states)
    
    draw_and_or_graph(states)

    conditional_plan = and_or_search(states)

    print("\nConditional plan:")
    pprint(conditional_plan)
