
"""
unfinished
"""

from numbers import Number
from operator import gt, ge, lt, le, ne, eq


class Constraint:
    _d = {'=': eq, '==': eq, '<>': ne, '!=': ne, '>': gt, '<': lt, '<=': le, '>=': ge}
    def __init__(self, variable1, variable2, relation):
        self.v1 = variable1
        self.v2 = variable2  # or number
        self.op = self._d.get(relation) or relation
        
        if hash(self.v1) == hash(self.v2):
            raise ValueError("Variables must be different!")
    
    def check(self, assignment=None):
        if assignment:
            v1 = assignment.get(self.v1)
            v2 = self.v2 if isinstance(self.v2, Number) else assignment.get(self.v2)
        else:
            v1 = self.v1.value
            v2 = self.v2 if isinstance(self.v2, Number) else self.v2.value
        if None in (v1,v2):
            return None   # or raise ValueError("No value assigned!")
        if not all(isinstance(v, Number) for v in (v1,v2)):
            raise TypeError("both must be numbers")
        return self.op(v1, v2)
    
    def __call__(self, assignment=None):
        return self.check(assignment)
    
    def __getitem__(self, index):
        return (self.v1, self.v2)[index]
    
    def __contains__(self, item):
        return (self[0] is item) or (self[1] is item)
    
    def __hash__(self):
        return round(hash(self.v1) + hash(self.v2) * 1.3 + hash(self.op))
    
    def __repr__(self):
        v1, v2 = (v if isinstance(v, Number) else v.name for v in (self.v1, self.v2))
        d = {v:k for k,v in self._d.items() if k not in ('==', '!=')}
        op = d[self.op]
        return self.__class__.__name__ + f"({v1}{op}{v2})"


       
class Variable:
    def __init__(self, name, domain=set(), value=None):
        self.name = name
        self.domain = domain
        self.value = value
    def __str__(self):
        return self.__class__.__name__ + f"({self.name})"
    def __repr__(self):
        return self.name
    def __hash__(self):
        return hash(self.name) + hash(tuple(self.domain))
    def __iter__(self):
        for e in self.domain:
            yield e
    def __len__(self):
        return len(self.domain)
    def __bool__(self):
        return len(self) > 0
    def __copy__(self):
        return self.__class__(self.name, self.domain)
    def copy(self):
        return self.__copy__()
    def eq(self, other):
        return self.name == other.name and frozenset(self.domain) == frozenset(other.domain) 
        
    def __gt__(self, other): return Constraint(self, other, relation='>')
    def __lt__(self, other): return Constraint(self, other, relation='<')
    def __eq__(self, other): return Constraint(self, other, relation='=')
    def __ne__(self, other): return Constraint(self, other, relation='!=')
    def __ge__(self, other): return Constraint(self, other, relation='>=')
    def __le__(self, other): return Constraint(self, other, relation='<=')
  
    

def check_assignment(assignment, constraints):
    for constraint in constraints:
        if constraint(assignment) is False:
            return False
    return True



def backtrack(variables, constraints):
    variables = set(variables)   # just in case
    
    def recurse(assignment):
        nonlocal variables, constraints
        # Base case
        if not variables:
            return assignment
        
        # Loop & recursive
        variable = variables.pop()
        
        for value in variable.domain:
            assignment[variable] = value
                        
            if check_assignment(assignment, constraints):
                result = recurse(assignment)   # result = {set of assignments} or False
                if result: return result
        else:  # if loop is over and no valid value is found
            del assignment[variable]
            variables.add(variable)
            return False
    return recurse(dict())



def revise(constraint):
    revised = False
    X,Y = constraint
    for x in tuple(X):  # tuple = copy
        X.value = x
        for y in (Y if isinstance(Y, Variable) else [Y]):
            if isinstance(Y, Variable): Y.value = y
            if constraint(): break
        else:
            X.domain.remove(x)
            revised = True
    return revised
            
            

def get_neighbors(variable, constraints, minus_variable=None):
    neighbors = set()
    for constraint in constraints:
        if minus_variable and (minus_variable in constraint):
            continue
        if variable in constraint:
            v1,v2 = constraint
            op = constraint.op
            if not((constraint[-1] is variable or isinstance(constraint[-1], Number))):
                v1,v2 = v2,v1
                op = {gt:lt, lt:gt, ge:le, le:ge}.get(op) or op
            neighbors.add(Constraint(v1, v2, op))
    return neighbors
            


def ac3(constraints):
    q = set(constraints)
    while q:
        constraint = q.pop()
        X,Y = constraint
        if revise(constraint):
            if not X: return False
            q = q.union(get_neighbors(X, constraints, minus_variable=Y))
    return True
            


def solve_assignment(variables, constraints):
    if not ac3(constraints):
        return None
    return backtrack(variables, constraints)
    
    
################################################################


A,B,C = (Variable(s, domain={1,2,3}) for s in "ABC")
variables = (A,B,C)
constraints = (A != 3, C < 4, A > B, B != C, A != C)
C.domain = {1,2,3,4}
solution = solve_assignment(variables, constraints)
print("solution:", solution)


X1 = Variable('X1', domain={1,2,3,4,5})
X2 = Variable('X2', domain={1,2,3})
X3 = Variable('X3', domain={3,4,5})
X4 = Variable('X4', domain={1,3,5,7})
variables = (X1,X2,X3,X4)
constraints = (X1<5, X1!=X2, X2>=X3, X4>X3, X4>5)
solution = solve_assignment(variables, constraints)
print("solution:", solution)


