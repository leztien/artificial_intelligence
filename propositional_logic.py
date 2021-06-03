"""
Logic - enumeration
"""


from operator import and_, or_, xor
from functools import reduce
from itertools import product


class Proposition:
    def __init__(self, str):
        self._proposition = str.lower().replace('.','')
    def __str__(self):
        return f"{self.__class__.__name__}({self._proposition})"
    def __repr__(self):
        return self.__str__()
    def evaluate(self, model):
        return bool(model[self])
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._proposition == other._proposition
    def __hash__(self):
        return hash(self._proposition)


class Not(Proposition):
    def __init__(self, proposition):
        self._proposition = proposition
    def evaluate(self, model):
        return not self._proposition.evaluate(model)
    def __iter__(self):
        yield self._proposition


class Connective(Proposition):
    def __init__(self, *propositions):
        self._propositions = propositions
        if len(set(self._propositions)) != len(self._propositions):
            raise ValueError("All propositions must be unique")
    def evaluate(self, model):
        return NotImplemented
    def __len__(self):
        return len(self._propositions)
    def __iter__(self):
        for e in tuple(self._propositions): yield e
    def __getitem__(self, index):
        return self._propositions[index]
    def __eq__(self, other):
        return isinstance(other, self.__class__) and len(self) == len(other) and all(p==q for p,q in zip(self,other))
    def __str__(self):
        return f"{self.__class__.__name__}({str(self._propositions)[1:-1]})"
    def __hash__(self):
        return hash(tuple(self))
    

class And(Connective):
    def evaluate(self, model):
        g = (p.evaluate(model) for p in self)
        return reduce(and_, g)
    
    
class Or(Connective):
    def evaluate(self, model):
        g = (p.evaluate(model) for p in self)
        return reduce(or_, g)


class Xor(Connective):
    def evaluate(self, model):
        g = (p.evaluate(model) for p in self)
        return reduce(xor, g)


class Implies(Connective):
    def _evaluate(self, p,q):
        model = self._temporary_model
        try:
            return not p.evaluate(model) or q.evaluate(model)
        except AttributeError:
            return not p or q.evaluate(model)
    def evaluate(self, model):
        self._temporary_model = model
        return reduce(self._evaluate, self)


class Iff(Connective):
    def evaluate(self, model):
        return reduce(lambda p,q: not xor(p,q),  (p.evaluate(model) for p in self))


def unravel_knowledge_base(kb):
    propositions = set()
    def _unravel(o):
        nonlocal propositions
        # Base case
        if type(o) is Proposition:
            propositions.add(o)
            return o
        # recursive case
        for e in o:
            _unravel(e)
    _unravel(kb)
    return propositions


def aks_knowledge_base(kb, query):
    propositions = tuple(unravel_knowledge_base(kb))
    results = []
    for permutation in product((0,1), repeat=len(propositions)):
        model = {k:v for k,v in zip(propositions, permutation)}
        if kb.evaluate(model):
            results.append(model[query])
    # Analize the results-list
    if len(set(results)) == 1:
        return bool(results[0])
    else:
        return "IDK"
            


#####################################################################

# EXAMPLE 1
"When ist is not raining Harry visits Hagrid."
"Harry visited Hagrid or Dumbledore but not both"
"Harry visited Dumbledore"
"Did it rain?"

p = Proposition("It is raining")
q = Proposition("Harry visist Hagrid")
r = Proposition("Harry visits Dumbledore")

kb = And(
        Implies(Not(p), q),
        Or(q, r),
        Not(And(q,r)),
        r)


ans = aks_knowledge_base(kb, query=p)
print("Did it rain?", ans)


# EXAMPLE 2
kb = And(Implies(And(p,q), r), p, q)
ans = aks_knowledge_base(kb, query=r)
print("Is the aircon on?", ans)


kb = Or(p,q)
ans = aks_knowledge_base(kb, query=p)
print("Did he visit Beijing?", ans)

kb = And(p,q)
ans = aks_knowledge_base(kb, query=p)
print("Did he visit Beijing?", ans)

# EXAMPLE 
p = Proposition("p")
q = Proposition("q")
r = Proposition("r")
s = Proposition("s")

kb = And(
    Iff(p,q),
    Implies(q, And(r,s)),
    p
    )

ans = aks_knowledge_base(kb, query=r)
print("r?", ans)


###

kb = And(
    Implies(p,q),
    Iff(q, Xor(r,s)),
    p, r
    )
ans = aks_knowledge_base(kb, query=s)
print("s?", ans)
