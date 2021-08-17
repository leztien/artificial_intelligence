

from enum import IntEnum



s = \
"""
§..§$.
....§.
.X..§.
.X....
"""

EMPTY = '.'
WALL = 'X'
GOAL = '$'
PITFALL = '§'

MAPPING = {EMPTY:-1, WALL:None, GOAL:+10, PITFALL:-10}
l = s.strip().split('\n')
grid = [[MAPPING[c] for c in s] for s in l]


states = {(i,j) for i in range(len(grid)) 
                for j in range(len(grid[0]))
                if grid[i][j] != MAPPING[WALL]}


class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

def actions(state):
    m,n = len(grid), len(grid[0])
    i,j = state
    assert (0 <= i < m and 0 <= j < n and grid[i][j] != MAPPING[WALL]), "bad arguments"
    
    if is_terminal(state):
        return set()
    
    candidates = {Actions.LEFT: (i, j-1),
                  Actions.RIGHT: (i, j+1),
                  Actions.UP: (i-1, j),
                  Actions.DOWN: (i+1, j)}
    actions = set()
    for action, (i,j) in candidates.items():
        if (0 <= i < m and 0 <= j < n and grid[i][j] != MAPPING[WALL]):
            actions.add(action)
            
    return actions


def transition_model(state, action):
    assert action in actions(state), "Bad action"
    i,j = state
    d = {Actions.LEFT: (i, j-1), Actions.RIGHT: (i, j+1),
               Actions.UP: (i-1, j), Actions.DOWN: (i+1, j)}
    return d[action]


def rewards(state, action):
    i,j = transition_model(state, action)
    return grid[i][j]


def is_terminal(state):
    assert state in states, "Bad argument"
    i,j = state
    r = grid[i][j]
    return (True if r == MAPPING[GOAL]
            or r == MAPPING[PITFALL]
            else False)


def make_policy(v_table):
    assert states == set(v_table), "error"
    return {state:{action: V[transition_model(state, action)] 
                   for action in actions(state)} 
                   for state in states}


def print_path(path, grid):
    grid = [[e for e in row] for row in grid]
    for square in path:
        i,j = square
        grid[i][j] = Ellipsis
    
    d = {Ellipsis:'*', MAPPING[EMPTY]: '.', MAPPING[WALL]: 'X', 
         MAPPING[GOAL]:'$', MAPPING[PITFALL]: '§'}
    print()
    for row in grid:
        for square in row:
            print(d[square], end='')
        print()
    print()
    
##############################################################################

# Hyperparameters
γ = 0.9


# Initialize V-table
V = {state:-.1 if not is_terminal(state) else grid[state[0]][state[1]]
     for state in states}



# Learning
for episode in range(1000):
    for s in states:
        values = []
        for a in actions(s):
            s_new = transition_model(s, a)
            r = rewards(s, a)
            v = r + γ*V[s_new]
            values.append(v)
        V[s] = max(values, default=0)

# Fill terminal states in V with the original values
for state in {s for s in states if is_terminal(s)}:
    i,j = state
    V[state] = grid[i][j]



# Inference
policy = make_policy(V)

start = (3,0)
state = start
path = [start,]

while not is_terminal(state):
    action = max(policy[state], key=lambda k: policy[state][k])
    state = transition_model(state, action)
    path.append(state)

print("path:", path)
print_path(path, grid)
