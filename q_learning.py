
"""
Q-Learning
"""


from random import choice, random



EMPTY = '.'
WALL = 'X'
GOAL = '+'
PITFALL = '-'

MAPPING = {EMPTY:-1, WALL:None, GOAL:+10, PITFALL:-10}


LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3



def make_grid(string):
    return [[MAPPING[c] for c in s] for s in s.strip().split('\n')]


def actions(state):
    m,n = len(grid), len(grid[0])
    i,j = state
    assert (0 <= i < m and 0 <= j < n and grid[i][j] != MAPPING[WALL]), "bad arguments"
    
    candidates = {LEFT: (i, j-1),
                  RIGHT: (i, j+1),
                  UP: (i-1, j),
                  DOWN: (i+1, j)}
    actions = []
    for action, (i,j) in candidates.items():
        if (0 <= i < m and 0 <= j < n and grid[i][j] != MAPPING[WALL]):
            actions.append(action)
    return actions


def transition_model(state, action):
    assert action in actions(state), "Bad action"
    i,j = state
    d = {LEFT: (i, j-1), RIGHT: (i, j+1), UP: (i-1, j), DOWN: (i+1, j)}
    return d[action]


def rewards(state, action):
    i,j = transition_model(state, action)
    return grid[i][j]


def is_terminal(state):
    assert state in STATES, "Bad argument"
    i,j = state
    r = grid[i][j]
    return True if r in (MAPPING[GOAL], MAPPING[PITFALL]) else False


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


def random_state(states=None):
    states = states or globals().get('STATES') or globals().get('states')
    while True:
        s = choice(states)
        if not is_terminal(s):
            return s


def select_action(state, epsilon=0.10):
    nx = actions(state)
    aa = [(i, Q[state][i]) for i in nx]
    if random() < epsilon:
        return choice(aa)[0]
    else:
        return max(aa, key=lambda t: t[-1])[0]


def infer_path(start, policy=None):
    policy = policy or globals().get('Q')
    path = [start,]
    state = start
    while not is_terminal(state):
        action = Q[state].index(max(Q[state]))
        state = transition_model(state, action)
        path.append(state)
    return path


##############################################################################

s = \
"""
-....+.
..X..-.
..X..-.
.......
"""


grid = make_grid(s)

STATES = sorted((i,j) for i in range(len(grid)) 
                for j in range(len(grid[0]))
                if grid[i][j] != MAPPING[WALL])
ACTIONS = (LEFT, RIGHT, UP, DOWN)


# Initialize the Q-table
Q = {state:[0]*len(ACTIONS) for state in STATES}

for state in Q:
    allowed_actions = actions(state)
    not_allowed = set(range(len(ACTIONS))) - set(allowed_actions)
    for ix in not_allowed:
        Q[state][ix] = -float('inf')


# Hyperparameters
γ = 0.9
η = 0.9


# Learning
for episode in range(1000):
    s = random_state()
    while not is_terminal(s):
        a = select_action(s)
        s_new = transition_model(s,a)
        r = rewards(s, a)
        TD = r + γ*max(Q[s_new]) - Q[s][a]
        Q[s][a] = Q[s][a] + η*TD
        s = s_new


# Inference
start = (3,0)
path = infer_path(start)
print_path(path, grid)
print("length of the path:", len(path))
