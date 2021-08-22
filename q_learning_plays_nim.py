
"""
Nim with Q-Learning
"""


from collections import UserDict
from random import choice, random


class State:
    def __init__(self, piles=[1,3,5,7]):
        if min(piles) < 0:
            raise ValueError("bad input")
        self.piles = piles
        self.player = 0
        self.winner = None
        
    @property
    def actions(self):
        return [(pile, j) for pile,n in enumerate(self.piles) for j in range(1, n+1)]
    
    def transition(self, action):
        pile, n = action
        if not((0 <= pile < len(self.piles)) and (1 <= n <= self.piles[pile])):
            raise ValueError("bad input")
        new = self.piles.copy()
        new[pile] -= n
        new = self.__class__(new)
        new.player = int(not self.player)
        new.winner = new.player if new.terminal else None
        return new
        
    @property
    def terminal(self):
        return max(self.piles) == 0
    def __str__(self):
        C = '*'
        return str.join('\n', (f"{i}: {C*n}" for i,n in enumerate(self.piles)))
    def __repr__(self):
        s = str(tuple(self.piles)).replace(', ', ',')
        return self.__class__.__name__ + s
    def __hash__(self):
        return hash(tuple(self.piles))
    def __eq__(self, other):
        return tuple(self.piles) == tuple(other.piles)


class QTable(UserDict):
    def __getitem__(self, index):
        if type(index) not in (State, tuple):
            raise TypeError("bad argument type")

        s,a = index if type(index) is tuple else (index, None)
        
        if s not in self:
            self.data[s] = {a:0 for a in s.actions}
        return self.data[s] if a is None else self.data[s][a]
    
    def __setitem__(self, index, item):
        if not (type(index) is tuple and type(item) in (int,float)):
            raise TypeError("bad arguments")
        s,a = index
        self[s][a] = item
    
    def get_best_value(self, state):
        return max(self[state].values())
    def get_best_action(self, state):
        return max(self[state], key=lambda action: self[state, action])
    def get_random_action(self, state):
        return choice(tuple(self[state]))


def policy(state, q_table, epsilon=0):
    # deterministic
    if not epsilon or random() < (1 - epsilon):
        return q_table.get_best_action(state)
    # stochastic
    return q_table.get_random_action(state)



#Q-Learning algorithm
epsilon = 0.1
lr = 0.5
gamma = 0.99

Q = QTable()

for episode in range(10000):
    s = State()
    prev = {0:None, 1:None}
    while True:
        a = policy(s, Q, epsilon)
        prev[s.player] = (s,a)
        s_ = s.transition(a)
        
        # If terminal state
        if s_.terminal:
            winner = s_.winner
            # Update the loser
            q_stored = Q[s,a]
            q_fresh = -1 + 0   # r = -1
            TD = q_fresh - q_stored
            Q[s,a] += lr*TD
            # Update the winner
            q_stored = Q[prev[s_.player]]
            q_fresh = +1 + 0   # r = +1
            TD = q_fresh - q_stored
            Q[prev[s_.player]] += lr*TD
            #Break out to the next episode
            break
        
        # If not terminal state
        elif prev[s_.player]:
            q_stored = Q[prev[s_.player]]
            r = 0
            v = Q.get_best_value(s_)
            q_fresh = r + gamma*v
            TD = q_fresh - q_stored
            Q[prev[s_.player]] += lr*TD
            
        # Reassign the new state
        s = s_
            

def get_input(prompt=None, state=None):
    s = input(prompt or "")
    i, *_, j = s
    i,j = int(i), int(j)
    
    if not state:
        return (i,j)
    
    while True:
        try:
            state.transition((i,j))
        except ValueError:
            print("Invalid input!")
            i,j = get_input(prompt)
            continue
        else:
            return (i,j)


def play(q_table):
    s = State()
    
    print("Play Nim!")
    print(s)
    
    while not s.terminal:
        # Human
        a = get_input("Human move: ", state=s)
        s = s.transition(a)
        print(s, "\n")
        
        if s.terminal:
            break
        
        # AI
        a = policy(s, q_table)
        s = s.transition(a)
        print("\nAI move:", a)
        print(s, "\n")
    
    # Win
    winner = "Human" if s.winner == 0 else "AI"
    print("Winner:", winner)
        
        

#############################################################
        
play(Q)

