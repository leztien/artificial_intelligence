
"""
Propositional Logic

Clue game
"""


from operator import and_, or_, xor
from functools import reduce
from itertools import product
from termcolor import cprint


class Proposition:
    def __init__(self, str):
        self._proposition = str.lower().replace('.','')
    def __repr__(self):
        return f"{self.__class__.__name__}({self._proposition})"
    def __str__(self):
        return self._proposition
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
        self._propositions = list(propositions)
        if any(not isinstance(e, Proposition) for e in propositions):
            raise TypeError("all elements must be of valid types: Proposition")
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
    def __repr__(self):
        return self.__str__()
    def __hash__(self):
        return hash(tuple(self))
    def add(self, item):
        if not isinstance(item, Proposition):
            raise TypeError("Must be of class Proposition" + f"\t({self.__class__.__name__})")
        self._propositions.append(item)
    def __add__(self, item):
        self.add(item)
        return self
    def __iadd__(self, item):
        self.add(item)
        return self

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


def entails(kb, query):
    """True = entails,   False = doesn't entail"""
    propositions = tuple(unravel_knowledge_base(kb))
    results = []
    for permutation in product((0,1), repeat=len(propositions)):
        model = {k:v for k,v in zip(propositions, permutation)}
        if kb.evaluate(model):
            results.append(query.evaluate(model))
    # Analize the results-list
    if len(set(results)) == 1 and results[0] == 1:
        return True
    else:
        return False # doesnt entail


def ask(kb, p):
    """Unlike entails() this function gives Yes/No/Maybe answer"""
    if entails(kb, p):
        return "Yes"
    elif not entails(kb, Not(p)):
        return "Maybe"
    else:
        return "No"


def solve_clue_game(knowledge, propositions):
    pad = 12
    d = {'Yes':'green', 'No':'red', 'Maybe':'blue'}
    for p in propositions:
        ans = ask(knowledge, p)
        cprint((str(p)+':').ljust(pad) + ans, d[ans])


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

entailment = entails(kb, query=p)
print("Did it rain?", entailment, end='\n\n')


# EXAMPLE 
p = Proposition("p")
q = Proposition("q")
r = Proposition("r")
s = Proposition("s")


kb = And(Xor(p,q), And(r,s))
ent = entails(kb, q)
ans = ask(kb, q)
print("entails:", ent, "\t answer:", ans, end='\n\n')


### THE 'CLUE' GAME ###
p1 = Proposition("ColMustard")
p2 = Proposition("ProfPlum")
p3 = Proposition("MsScarlet")
characters = [p1,p2,p3]

r1 = Proposition("ballroom")
r2 = Proposition("kitchen")
r3 = Proposition("library")
rooms = [r1,r2,r3]

w1 = Proposition("knife")
w2 = Proposition("revolver")
w3 = Proposition("wrench")
weapons = [w1,w2,w3]

propositions = characters + rooms + weapons

# Given by the rules of the game:
kb = And(Or(p1,p2,p3),
         Or(r1,r2,r3),
         Or(w1,w2,w3))

# One card is drawn
kb.add(Not(p1))


# Some other cards are drawn
kb.add(Not(p2))
kb.add(Not(r2))
kb.add(Not(r1))
kb.add(Not(w2))

# A player makes a (wrong) guess
kb += Not(And(p2, r2, w2))

# A second player makes a wrong guess
kb = kb + Or(Not(p3), Not(r2), Not(w2))

# Finaly a third pplayer makes a wrong guess at which stage the system gueses correctly
kb += Or(Not(p3), Not(r3), Not(w3))  # COMMENT THIS LINE OUT TO SEE THE DIFFERENCE

# Attemp a solution
solve_clue_game(knowledge=kb, propositions=propositions)

