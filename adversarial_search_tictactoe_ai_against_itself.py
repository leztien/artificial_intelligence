
"""
AI plays Tic-Tac_toe against itself.
Adverserial Search with Tic-Tac-Toe
"""

from typing import List, Tuple
from enum import Enum
from math import inf
from copy import deepcopy
from random import randint, random, choice


class Player(Enum):
    X = 'X'   # MAX-player
    O = 'O'   # MIN-player
    EMPTY = chr(0x25a1)



class State():    
    def __init__(self, *args, **kwargs):
        self._mx = list(args[0]) if args else [[Player.EMPTY,]*3 for _ in range(3)]
        self.utility_value = kwargs.get("utility")   # None by default
        self._terminal = None
        
        # Check state validity
        l = sum(self._mx, [])
        xx,oo = (l.count(v) for v in (Player.X, Player.O))
        if not(0 <= (xx - oo) <=1):
            raise ValueError("Invalid board state")
        
    def player(self) -> Player:
        """Returns the player who's turn it is"""
        l = sum(self._mx, [])
        xx,oo = (l.count(v) for v in (Player.X, Player.O))
        return Player.X if xx - oo == 0 else Player.O
    
    def actions(self) -> List[Tuple[int,int]]:
        """Returns valid moves from this state"""
        return [(i,j) for i in range(3) for j in range(3) if self._mx[i][j] == Player.EMPTY]
    
    def result(self, action) -> 'State':
        """returns the next state"""
        assert min(action) >= 0 and max(action) <= 2
        copy = deepcopy(self._mx)
        r,c = action
        if copy[r][c] != Player.EMPTY:
            raise ValueError("bad action (tuple-index)")
        player = self.player()
        copy[r][c] = player
        s = State(copy)
        r,c = [(i,j) for i in range(3) for j in range(3) if self._mx[i][j] != s._mx[i][j]][0]
        s.move = "ABC"[r] + str(c+1)
        return s
    
    def children(self) -> List:
        """returns next states from this state.
        This is a wrapper-function around actions() and result() methods"""
        return [self.result(action) for action in self.actions()]
    
    @property
    def terminal(self) -> bool:
        """is this state terminal?"""
        if self.utility():
            return True
        
        l = sum(self._mx, [])
        slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
        
        for e in self._mx + slices:
            S = set(l[e] if isinstance(e, slice) else e)
            if not {Player.X, Player.O}.issubset(S) and Player.EMPTY in S:
                return False
        return True
    
    def utility(self, depth=None) -> int or float:
        """returns the utility"""
        l = sum(self._mx, [])
        slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
        depth = abs(depth or 1)
        
        for e in self._mx + slices:
            t = l[e] if isinstance(e, slice) else e
            if t.count(Player.X) == 3:
                return 1 / depth
            elif t.count(Player.O) == 3:
                return -1 / depth
        return 0  
        
    def __str__(self):
        return str.join('\n', ((str.join('', (v.value for v in row))) for row in self._mx))
    
    def __lt__(self, other):
        return self.utility_value < (other.utility_value if isinstance(other, State) else other)
    def __gt__(self, other):
        return self.utility_value > (other.utility_value if isinstance(other, State) else other)



def min_play(state, depth=0):
    # Base case
    if state.terminal:
        if depth==0: return state
        return state.utility(depth=depth)
    # Recursive
    best = inf
    for child in state.children():
        v = max_play(child, depth-1)
        if depth==0:
            child.utility_value = v
            v = child
        best = min(v, best)
    
    # Figure out what move the ai has made for printing purposes
    if depth == 0 and isinstance(best, State):
        r,c = [(i,j) for i in range(3) for j in range(3) if state._mx[i][j] != best._mx[i][j]][0]
        best.move = "ABC"[r] + str(c+1)
    return best


def max_play(state, depth=0):
    # Base case
    if state.terminal:
        if depth==0: return state
        return state.utility(depth=depth)
    # Recursive
    best = -inf
    for child in state.children():
        v = min_play(child, depth-1)
        if depth==0:
            child.utility_value = v
            v = child
        best = max(v, best)
        
    # Figure out what move the ai has made for printing purposes
    if depth == 0 and isinstance(best, State):
        r,c = [(i,j) for i in range(3) for j in range(3) if state._mx[i][j] != best._mx[i][j]][0]
        best.move = "ABC"[r] + str(c+1)
    return best


def decorator(func):
    """Checks whether the square chosen by the user is occupied"""
    def closure(*args, **kwargs):
        state = kwargs.get("state")
        actions = kwargs.get("actions")
        
        while True:
            user_input = func()
            if not(state or actions):
                return user_input
            
            # Get allowed actions (i.e. moves) - if not provided by the user
            actions = actions or state.actions()
            
            # Check validity of the move
            if user_input not in actions:
                print("The selected square is taken.")
            else:
                return user_input
    return closure


@decorator
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
    if state.terminal:
        print(state)
        cost = state.utility()
        winner = {0:None, 1:Player.X, -1:Player.O}[cost]
        if not winner:
            print("\nDRAW")
        else:
            print("\nWINNER:", winner.value)
        return winner or "DRAW"
    return False


def controller(randomness=True):
    """Simulates two AI's playing against each other.
    X starts.  if randomness=True then X makes sometines random moves
    which enables O to win at times.
    Otherwise each game ends in a draw"""
    
    # Initial state
    s = State()
    action = [randint(0,2) for _ in range(2)]
    s = s.result(action)   # X player
    print("\nX's move:", s, sep='\n', end='\n\n')
    
    while True:    
        # O's move
        s = min_play(s)   # s = AI's move
        print(f"O's move: {s.move}")
        print(s, end='\n\n')
        
        winner = check_winner(s)
        if winner: break
        
        # X's move
        if randomness and random() < 0.35:
            a = choice(s.actions()) # random allowed action
            s = s.result(a)
            print(f"X's random move: {s.move}")
        else:
            s = max_play(s)    
            print(f"X's move: {s.move}")
            
        print(s, end='\n\n')
        
        winner = check_winner(s)
        if winner: break
        
    return winner

    
    

winner = controller(randomness=True)
print("The winner is:", winner)

