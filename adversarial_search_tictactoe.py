
"""
Adverserial Search with Tic-Tac-Toe
"""

from enum import Enum
from math import inf



class Player(Enum):
    X = 'X'   # MAX-player
    O = 'O'   # MIN-player
    EMPTY = chr(0x25a1)




class State():    
    def __init__(self, *args, **kwargs):
        self._mx = list(args[0]) if args else [[Player.EMPTY,]*3 for _ in range(3)]
        # Check state validity
        l = sum(self._mx, [])
        xx,oo = (l.count(v) for v in (Player.X, Player.O))
        assert 0 <= (xx - oo) <=1, "invalid state"
        self.utility = None
    
    def __str__(self):
        return str.join('\n', ((str.join('', (v.value for v in row))) for row in self._mx))
    
    def __lt__(self, other):
        return self.utility < (other.utility if isinstance(other, State) else other)
    def __gt__(self, other):
        return self.utility > (other.utility if isinstance(other, State) else other)




def get_player(state):
    l = sum(state._mx, [])
    xx,oo = (l.count(v) for v in (Player.X, Player.O))
    return Player.X if xx - oo == 0 else Player.O
    

def get_actions(state):
    """returns a list of tuples"""
    return [(i,j) for i in range(3) for j in range(3) if state._mx[i][j] == Player.EMPTY]


def get_result(state, action) -> State:
    assert min(action) >= 0 and max(action) <= 2
    copy = [[v for v in row] for row in state._mx]
    r,c = action
    if copy[r][c] != Player.EMPTY:
        raise ValueError("bad action (tuple-index)")
    player = get_player(state)
    copy[r][c] = player
    return State(copy)
    


def is_terminal(state) -> bool:
    if utility(state):
        return True
    
    l = sum(state._mx, [])
    slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
    
    for e in state._mx + slices:
        S = set(l[e] if isinstance(e, slice) else e)
        if not {Player.X, Player.O}.issubset(S) and Player.EMPTY in S:
            return False
    return True
        
    


def utility(state, depth=None) -> int:
    l = sum(state._mx, [])
    slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
    depth = abs(depth or 1)
    
    for e in state._mx + slices:
        t = l[e] if isinstance(e, slice) else e
        if t.count(Player.X) == 3:
            return 1 / depth
        elif t.count(Player.O) == 3:
            return -1 / depth
    return 0            
    
    

def minimax(state, maximizing=False, depth=0):
    # Base case
    if is_terminal(state):
        state.utility = utility(state, depth=depth)
        if depth==0: return state
        else: return state.utility
    
    # Recursion
    actions = get_actions(state)
    children = [get_result(state, action) for action in actions]
    
    if maximizing:
        best = -inf
        for child in children:
            v = minimax(child, maximizing=False, depth=depth-1)
            if depth==0:
                child.utility = v
                v = child
            best = max(v, best)
    else:
        best = inf
        for child in children:
            v = minimax(child, maximizing=True, depth=depth-1)
            if depth==0:
                child.utility = v
                v = child
            best = min(v, best)
    # Figure out what move the ai has made for printing purposes
    if depth == 0 and isinstance(best, State):
        r,c = [(i,j) for i in range(3) for j in range(3) if state._mx[i][j] != best._mx[i][j]][0]
        move = "ABC"[r] + str(c+1)
        best.move = move
    return best
        

    
def wrapper(func):
    """Checks whether the square chosen by the user is occupied"""
    def closure(*args, **kwargs):
        state = kwargs.get("state")
        actions = kwargs.get("actions")
        
        while True:
            user_input = func()
            if not(state or actions):
                return user_input
            
            # Get allowed actions (i.e. moves) - if not provided by the user
            actions = actions or get_actions(state)
            
            # Check validity of the move
            if user_input not in actions:
                print("The selected square is taken.")
            else:
                return user_input
    return closure


@wrapper
def get_input():
    rows = {'a':0, 'b':1, 'c':2}
    cols = {str(i+1):i for i in range(3)}
    while True:
        s = input("Your move: ")
        s = s.lower().strip()
        if len(s)!=2 or s[0] not in rows.keys() or s[1] not in cols.keys():
            print("bad input - accpted format: A1")
            continue
        else:
            return (rows[s[0]], cols[s[1]])

    

def check_winner(state):
    if is_terminal(state):
        print(state)
        cost = utility(state)
        winner = {0:None, 1:Player.X, -1:Player.O}[cost]
        if not winner:
            print("\nDRAW")
        else:
            print("\nWINNER:", winner.value)
        return winner or "DRAW"
    return False


def controller():
    """the human starts the game as X"""
    
    # Initial state
    s = State()
    print("\nTIC TAC TOE", s, sep='\n')
    
    while True:    
        # Human's move
        a = get_input(state=s)
        s = get_result(s,a)
        
        winner = check_winner(s)
        if winner: break
        
        # AI's move
        s = minimax(s)   # s = AI's move
        print(f"My move: {s.move}")
        
        winner = check_winner(s)
        if winner: break
        
        print(s)
    return winner

    
    

winner = controller()
print("The winner is:", winner)
