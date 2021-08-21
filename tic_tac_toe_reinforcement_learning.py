"""
Q-Learning
MiniMax
(2 in 1)
"""


from typing import List, Tuple
from enum import Enum
from math import inf
from copy import deepcopy
from collections import UserDict
from random import random, choice



class Player(Enum):
    X = 'X'   # MAX-player
    O = 'O'   # MIN-player
    EMPTY = chr(0x25a1)
    DRAW = "DRAW"



class State():
    gamma = 0.9
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
    
    def transition(self, action) -> 'State':
        """returns the next state"""
        assert min(action) >= 0 and max(action) <= 2
        copy = deepcopy(self._mx)
        r,c = action
        if copy[r][c] != Player.EMPTY:
            raise ValueError("bad action (tuple-index)")
        player = self.player()
        copy[r][c] = player
        
        new = State(copy)
        
        # for printing purposes
        r,c = [(i,j) for i in range(3) for j in range(3) if self._mx[i][j] != new._mx[i][j]][0]
        new.move = "ABC"[r] + str(c+1)
        
        return new
    
    def children(self) -> List:
        """returns next states from this state.
        This is a wrapper-function around actions() and transition() methods"""
        return [self.transition(action) for action in self.actions()]
    
    @property
    def terminal(self) -> bool:
        """is this state terminal?"""
        if self.utility():
            return True
        
        l = sum(self._mx, [])
        
        # This line is for RL (play to the last square, even when draw)
        return False if Player.EMPTY in l else True
        
        # This block is for Adversarial Search (returns True when it is clear that it is a draw)
        slices = [slice(i,None,3) for i in range(3)] + [slice(None,None,4), slice(2,-1,2)]
        
        for e in self._mx + slices:
            S = set(l[e] if isinstance(e, slice) else e)
            if not {Player.X, Player.O}.issubset(S) and Player.EMPTY in S:
                return False
        return True
    
    
    def winner(self):
        if not self.terminal:
            return None
        cost = self.utility()
        winner = {0:Player.DRAW, 1:Player.X, -1:Player.O}[cost]
        return winner
        
    
    def utility(self, depth=None) -> int or float:
        """returns the utility (used in Adverserial Search)"""
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
    
    def __repr__(self):
        d = {Player.X: [], Player.O: []}
        for i,p in enumerate(sum(self._mx, [])):
            if p == Player.EMPTY:
                continue
            d[p].append(i+1)
            
        s = 'X:' + str(d[Player.X])[1:-1].replace(' ', '') + " / O:" + str(d[Player.O])[1:-1].replace(' ', '')
        return self.__class__.__name__ + f"({s})"
    
    def __lt__(self, other):
        return self.utility_value < (other.utility_value if isinstance(other, State) else other)
    def __gt__(self, other):
        return self.utility_value > (other.utility_value if isinstance(other, State) else other)
    
    def __hash__(self):
        return hash(tuple(sum(self._mx, [])))
    def __eq__(self, other):
        if type(self) is not type(other):
            raise TypeError(f"equality operation is not supported between '{self.__class__.__name__}' and '{other.__class__.__name__}'")
        return hash(self) == hash(other)
    
    @property
    def user_friendly_board(self):
        l = ["  123"] + [c+' '+line for c,line in zip("ABC", str(self).strip().split('\n'))]
        return str.join('\n', l)
    
    def reward(self, action=None):
        """returns immediate reward from the standpoint of player O"""
        d = {Player.X: -100, Player.O: 100, Player.DRAW: 0, None: 0}
        state = self if not action else self.transition(action)
        winner = state.winner()
        return d[winner]
                
    def value(self):
        """returns the value of the state (recursive function with a base case)"""
        if self.terminal:
            return self.reward()
        values = (s.value() for s in self.children())
        return self.reward() + self.gamma * max(values)
        
        
class QTable(UserDict):
    def __getitem__(self, index):
        if type(index) not in (State, tuple):
            raise TypeError("bad argument type")

        s,a = index if type(index) is tuple else (index, None)
        
        if s not in self:
            self.data[s] = {a:0 for a in s.actions()}
        return self.data[s] if a is None else self.data[s][a]
    
    def __setitem__(self, index, item):
        if not (type(index) is tuple and type(item) in (int,float)):
            raise TypeError("bad arguments")
        s,a = index
        self[s][a] = item


def policy(state, Q_table, epsilon=0):
    # deterministic
    if not epsilon or random() < (1 - epsilon):
        return max(Q_table[state], key=lambda k: Q_table[state][k])
    # stochastic
    return choice(tuple(Q_table[state]))


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
    print(">>>WINNER:", state.winner())
    
    if state.terminal:
        print(state.user_friendly_board)
        cost = state.utility()
        winner = {0:None, 1:Player.X, -1:Player.O}[cost]
        if not winner:
            print("\nDRAW")
        else:
            print("\nWINNER:", winner.value)
        return winner or "DRAW"
    return False


def controller(player=None, Q_table=None):
    """
    Controlls the game.
    The AI is able to play in two modes: MiniMax or Q-Learning.
    
    If player=None  then the user is asked.
    """
    
    print("\nYour AI opponent: " + ("RL" if Q_table else "MiniMax"))
    
    human_as_X = None if player is None else True if 'X' in str(player).upper() else False
    
    if human_as_X is None:
        print("Do you want to play as 'X' or 'O'?\t\t(the 'X' starts the game)")
        answer = input("X or O? ")[-1].upper(); print()
        human_as_X = 'X' == answer

    
    # Initial state
    s = State()
    
    # If the human goes first
    if human_as_X:
        print(s.user_friendly_board)
        a = get_input(state=s)
        s = s.transition(a)
        print(s.user_friendly_board)
    
    # The main loop
    while True:
        # AI's move
        if Q_table:
            a = policy(s, Q)
            s = s.transition(a)
        else:
            s = (min_play if human_as_X else max_play)(s)

        print(f"\nMy move: {s.move}")
        print(s.user_friendly_board)
        
        winner = s.winner()
        if winner: break
        
    
        # Human's move
        a = get_input(state=s)
        s = s.transition(a)
        
        print(s.user_friendly_board)
        
        winner = s.winner()
        if winner: break
    
    print("\nThe winner is: " + winner.value)
    return winner

#_____________________________________________________________________________

## Q-learning
η = 0.9
γ = 0.9



r = 1
r_draw = 0.2
γ = 0.9
ε = 0.3
η = 0.25

Q = QTable()

for episode in range(100000):
    s = State()
    path = []
    
    while not s.terminal:
        a = policy(s, Q, ε)
        path.append((s,a))
        s = s.transition(a)
    
    # Who's the winner?
    winner = s.winner()
        
    #
    for (s,a) in path[::-1]:
        if winner == Player.DRAW:
            TD = r_draw - Q[s,a]
        elif s.player() == winner:
            TD = r - Q[s,a]
            #Q[s,a] += r
        else:
            TD = -r - Q[s,a]
            #Q[s,a] -= r
        Q[s,a] = Q[s,a] + η * TD
        
        # doscount the reward
        r *= γ
        r_draw *= γ
    
#______________________________________________________

winner = controller(Q_table=Q)

