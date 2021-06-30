"""
Adverserial Search with Tic-Tac-Toe
- with minimax returning numerical values and find_best_move() function that returns the best state
- State class encapsulates all relevant attributes and funcions(i.e. methods)
"""

from typing import List, Tuple
from enum import Enum
from math import inf
from copy import deepcopy



class Player(Enum):
    X = 'X'   # MAX-player
    O = 'O'   # MIN-player
    EMPTY = chr(0x25a1)



class State():    
    def __init__(self, *args, **kwargs):
        self._mx = list(args[0]) if args else [[Player.EMPTY,]*3 for _ in range(3)]
        self._utility = kwargs.get("utility", None)   # None by default
        self._terminal = None
        self._player = None
        self._actions = None
        
        # Check state validity
        l = sum(self._mx, [])
        xx,oo = (l.count(v) for v in (Player.X, Player.O))
        if not(0 <= (xx - oo) <=1):
            raise ValueError("Invalid board state")
    
    @property
    def player(self) -> Player:
        """Returns the player who's turn it is"""
        xx,oo = (sum(self._mx, []).count(v) for v in (Player.X, Player.O))
        self._player = Player.X if (xx - oo) == 0 else Player.O
        return self._player
    
    @property
    def turn(self):
        ...
        
    @property
    def actions(self) -> List[Tuple[int,int]]:
        """Returns valid moves from this state"""
        self._actions = [(i,j) for i in range(3) for j in range(3) if self._mx[i][j] == Player.EMPTY]
        return self._actions
        
    def result(self, action) -> 'State':
        """returns the next state"""
        assert min(action) >= 0 and max(action) <= 2
        copy = deepcopy(self._mx)
        r,c = action
        if copy[r][c] != Player.EMPTY:
            raise ValueError("bad action (tuple-index)")
        copy[r][c] = self.player
        s = State(copy)
        
        # attach move in str-form for printing purposes
        r,c = [(i,j) for i in range(3) for j in range(3) if self._mx[i][j] != s._mx[i][j]][0]
        s.move = "ABC"[r] + str(c+1)
        return s
    
    def children(self) -> List:
        """returns next states from this state.
        This is a wrapper-function around actions() and result() methods"""
        return [self.result(action) for action in self.actions]
    
    @property
    def terminal(self) -> bool:
        """is this state terminal?"""
        if self.utility:
            return True
        
        l = sum(self._mx, [])
        slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
        
        for e in self._mx + slices:
            S = set(l[e] if isinstance(e, slice) else e)
            if not {Player.X, Player.O}.issubset(S) and Player.EMPTY in S:
                self._terminal = False
                return self._terminal
        self._terminal = True
        return self._terminal
    
    @property
    def utility(self) -> int or float:
        """returns the utility"""
        if self._utility:
            return self._utility
        # Else:  calculate and return
        l = sum(self._mx, [])
        slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
        
        for e in self._mx + slices:
            t = l[e] if isinstance(e, slice) else e
            if t.count(Player.X) == 3:
                return 1
            elif t.count(Player.O) == 3:
                return -1
        return 0  
    
    @utility.setter
    def utility(self, value):
        self._utility = value
    
    def __str__(self):
        return str.join('\n', ((str.join('', (v.value for v in row))) for row in self._mx))
    
    def __lt__(self, other):
        return self.utility < (other.utility if isinstance(other, State) else other)
    def __gt__(self, other):
        return self.utility > (other.utility if isinstance(other, State) else other)



def minimax(state, maximizing):
    # Base case
    if state.terminal:
        return state.utility
    # Recusrive cases
    best = -inf if maximizing else inf
    if maximizing:
        for action in state.actions:
            child = state.result(action)
            v = minimax(child, maximizing=False)
            best = max(v, best)
    else:
        for child in state.children():  # slightly different implementation
            v = minimax(child, maximizing=True)
            best = min(v, best)
    return best
    
    
def find_best_move(state):
    """Retunrs the best action/state for the MIN player"""
    best = inf
    for s in state.children():
        s.utility = minimax(s, maximizing=True)
        best = min(s, best)   # best is a State()
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
            actions = actions or state.actions
            
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
        cost = state.utility
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
    print("\nYou play as X", s, sep='\n')
    
    while True:    
        # Human's move
        a = get_input(state=s)
        s = s.result(a)
        
        winner = check_winner(s)
        if winner: break
        
        # AI's move
        s = find_best_move(s)   # s = AI's move
        print(f"My move: {s.move}")
        
        winner = check_winner(s)
        if winner: break
        
        print(s)
    return winner

    
    

winner = controller()
print("The winner is:", winner)
