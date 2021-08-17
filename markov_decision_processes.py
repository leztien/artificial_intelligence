"""
Markov Decision Processes
https://www.youtube.com/watch?v=KovN7WKI9Y0
"""


STATES = A,B,C = range(3)
ACTIONS = 0,1


REWARDS = [10, -1, -10]

U = {A: 0.0, B: 0.0, C: 0.0}
U = {state: 0.0 for state in STATES}


P = {
(A,0):{A: 1.0, B: 0, C:0},
(B,0):{A: 0.8, B:0.2, C:0},
(C,0):{A:1, B:0, C:0},
(A,1):{A: 0.1, B: 0.9, C:0},
(B,1):{A:0, B:0.3, C:0.7},
(C,1):{A:0, B:0, C:1}
     }


gamma = 0.99



def updated_utility(s):
    return REWARDS[s] +\
        gamma * max(sum(P[s,a][s_] * U[s_] for s_ in STATES) for a in ACTIONS)


for i in range(10000):
    copy = U.copy()
    for state in STATES:
        U[state] = updated_utility(state)
    if U == copy:
        print(f"converged after {i} iterations")
        break
print(U)


for s in STATES:
    assert U[s] == updated_utility(s)
