
"""
Eight Queens
"""

from collections import defaultdict
from random import shuffle


class Checkerboard:
    def __init__(self, k):
        self._mx = [[0]*k for _ in range(k)]
    def __getitem__(self, index):
        try: i,j = index
        except TypeError: return self._mx[index]
        else: return self._mx[i][j]
    def __setitem__(self, index, value):
        i,j = index
        self._mx[i][j] = value
    def __repr__(self):
        return self._mx.__str__()[1:-1].replace('], [', ']\n[')
    def __str__(self):
        d = {0:chr(0x25A2), 1:chr(0x265B)}
        return str.join('\n', (str.join('', (d[e] for e in row)) for row in self))
    def __len__(self):
        return len(self._mx)
    def __bool__(self):
        return sum(sum(self, []))==len(self) and ok(self)


def ok(M):
    """
    are none of the queens so far threatened (by any other)?
    """
    k = len(M)
    # Check row threat
    for row in M:
        if sum(row) > 1:
            return False
    # Check column threat
    a = sum(M, [])
    for i in range(k):
        col = a[i::k]
        if sum(col) > 1:
            return False
    # Check diagonal
    d = defaultdict(list)
    for i in range(k):
        for j in range(k):
            d[i-j].append(M[i,j])
            d[i+j+k].append(M[i,j])
    for l in d.values():
        if sum(l) > 1:
            return False
    # Otherwise ok
    return True
    
    
def queens(M, i=0):
    k = len(M)
    
    # Base
    if i == k:
        return M
    # Recursive
    nx = list(range(k))
    shuffle(nx)
    
    for j in nx:
        M[i,j] = 1
        if ok(M):
            result = queens(M, i+1)
            if result:
                return result
        M[i,j] = 0
    # None of the tried squares in this row is ok - return False i.e. get to the previous row
    return False

###########################################################

k = 8
mx = Checkerboard(k)

mx = queens(mx)
print("solved queens:\n", mx, sep='')
print("final check:", ok(mx), bool(mx))
