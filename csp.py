

import random
from string import ascii_uppercase
from operator import ne, eq, gt, lt, ge, le
from collections import namedtuple
from copy import deepcopy


csp_namedtuple = namedtuple(typename="csp", field_names=["variables", "constraints", "solution"])


def make_csp(variables, domains=None, arcs=None, different_domains=False,
             return_all_solutions=False, random_seed=False):
    """
    variables: int or iterable containing the (names of) variables
    domains : None or iterable (one domain for all or individual domains for each variable)
    arcs : float denotes the percentage/density of arcs
           'ne' denotes 'not equal' arcs for all arcs
    return_all_solutions: if True, attempts to return all solutions
    """

    d = dict()
    
    if type(variables) is int:
        if variables <= 26:
            variables = tuple(ascii_uppercase[:variables])
        else:
            variables = [f'X{i}' for i in range(1, variables+1)]
    else:
        variables = tuple(variables)
    if len(variables) < 3:
        raise ValueError("The number of variables must be at least 3")
    
    # Check domains arg
    if domains is None:
        domains = set(range(len(variables)))    
        
    if not (domains and hasattr(domains, '__iter__') and type(domains) is not str):
        raise ValueError("'domains' must be a valid iterable")
    
    if not hasattr(tuple(domains)[0], '__iter__'):
        d = {k:set(domains) for k in variables}
    else:
        if len(variables) != len(domains):
            raise ValueError("inconsistent number of variables and domains")
        d = {k:set(v) for k,v in zip(variables, domains)}
        #assert all(len(e)>=2 for e in d.values()), "one or more domains less than 2"
    
    # Random seed
    if random_seed is True:
        random_seed = random.randint(100,999)
        print("random seed:", random_seed)
    if type(random_seed) is int:
        random.seed(random_seed)
    
    #Different domains?
    PROB = 0.5
    if different_domains and len(set(tuple(e) for e in d.values()))==1:
        for k in d:
            if random.random() < PROB and len(d[k]) >= 3:
                v = random.choice(tuple(d[k]))
                d[k].remove(v)
            
    
    
    
    # Make a solution
    solution = {k: random.choice(tuple(v)) for k,v in d.items()}
    
    #...
    variables = list(solution.keys())
    constraints = []
    
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            X,Y = variables[i], variables[j]
            if arcs == 'ne':
                if solution[X] != solution[Y]:
                    constraints.append((X, Y, '<>'))
                continue
            # If not ne
            if solution[X] == solution[Y]:
                constraints.append((X,Y,'='))
            elif solution[X] == solution[Y]+1:
                constraints.append((X,Y,'>='))
            elif solution[X]+1 == solution[Y]:
                constraints.append((X,Y,'<='))
            elif solution[X] < solution[Y]:
                constraints.append((X,Y,'<'))
            else:
                constraints.append((X,Y,'>'))
    
    if type(arcs) is float:
        random.shuffle(constraints)
        constraints = constraints[:round(len(constraints) * arcs)]
    
    if return_all_solutions:
        try:
            from constraint import Problem
        except ImportError:
            print("unable to import the 'constraint' library")
            solution = [solution]
        else:
            funcs = {'<':lt, '>':gt, '==':eq, '=':eq, '<>':ne, '!=':ne, '<=':le, '>=':ge}
            
            csp = Problem()
            
            for var,dom in d.items():
                csp.addVariable(var, list(dom))
            
            for C in constraints:
                csp.addConstraint(funcs[C[-1]], (C[:2]))
            
            assert solution in csp.getSolutions(), "error"
            solution = csp.getSolutions()
    
    func = namedtuple(typename="csp", field_names=["variables", "constraints", "solution"])
    return func(d, constraints, solution)


def draw_graph(csp:namedtuple):
    import networkx as nx
    import matplotlib.pyplot as plt
    
    variables = csp.variables.keys()
    constraints = csp.constraints
    
    G = nx.Graph()
    
    G.add_nodes_from(variables)
    
    edge_labels = dict()
    
    for C in constraints:
        G.add_edge(*C[:2])
        edge_labels[(C[:2])] = C[-1]

    pos = nx.spring_layout(G)
    
    plt.figure()  
    
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=500,node_color='pink',alpha=0.9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.axis('off')
    plt.show()

#_____________________________________________________________________________


def select_unassigned_variable(assignment, csp):
    unassigned = set(csp.variables.keys()) - set(assignment.keys())
    
    # MRV (min values)
    mrv = [(v, len(csp.variables[v])) for v in unassigned]
    mn = min(mrv, key=lambda t: t[-1])[1]
    unassigned = [t[0] for t in mrv if t[-1] <= mn]
    
    # Degrees 8max value)
    degrees = [(X, sum(X in arc for arc in csp.constraints)) for X in unassigned]
    return max(degrees, key=lambda t: t[-1])[0]


def order_domain_values(variable, csp):
    values = sorted(csp.variables[variable])
    return values


def consistent(assignment, constraints):
    ops = {'<':lt, '>':gt, '==':eq, '=':eq, '<>':ne, '!=':ne, '<=':le, '>=':ge}    
    for constraint in constraints:
        x,y = assignment.get(constraint[0]), assignment.get(constraint[1])
        if None in (x,y):
            continue
        op = ops[constraint[-1]]
        if not op(x,y):
            return False    
    return True


def inference(variable, assignment, csp):
    
    X = variable
    constraints = [c for c in csp.constraints if c[1] == X]
    
    variables = {k: {assignment[k]} if assignment.get(k) else csp.variables[k] for k in csp.variables.keys()}
    
    csp = csp_namedtuple(variables=variables, constraints=constraints, solution=csp.solution)    
    csp = ac3(csp)
    
    if not csp or csp.variables == variables:
        return dict()
    
    inferences = dict()
    
    for X, dom in csp.variables.items():
        if X not in assignment and len(dom)==1:
            inferences[X] = dom.pop()
    
    return inferences


def backtrack(assignment, csp):
    # Base case
    if len(assignment) == len(csp.variables):
        return assignment
    # Recursive case
    X = select_unassigned_variable(assignment, csp)
    for x in order_domain_values(X, csp):
        d = assignment.copy()
        d[X] = x
        if consistent(d, csp.constraints):
            inferences = inference(X, d, csp)
            d.update(inferences)
            solution = backtrack(d, csp)
            if solution:
                return solution
    #for-loop ended
    return False


def remove_inconsistent_values(constraint, csp):
    """
    the 'revise' function
    """
    ops = {'<':lt, '>':gt, '==':eq, '=':eq, '<>':ne, '!=':ne, '<=':le, '>=':ge}
    X,Y,op = constraint
    op = ops[op]
    X_domain = csp.variables[X]
    Y_domain = csp.variables[Y]
    
    removed = False
    
    for x in X_domain.copy():
        for y in Y_domain:
            if op(x,y):
                break
        else:
            X_domain.remove(x)
            removed = True

    return removed


def make_all_arcs(csp):
    ops = {'<':'>', '>':'<', '=':'=', '==':'==', '<>':'<>', '!=':'!=', '>=':'<=', '<=':'>='}
    agenda = list(csp.constraints)
    for constraint in csp.constraints:
        agenda.append((constraint[1], constraint[0], ops[constraint[-1]]))
    return agenda


def get_neighbors(constraint, arcs):
    X,Y,_ = constraint
    neighbors = set()
    for arc in arcs:
        if arc[1] == X and arc[0] != Y:
            neighbors.add(arc)
    return neighbors


def ac3(csp):
    """
    returns a new csp-namedtuple
    """
    csp = csp_namedtuple(variables=deepcopy(csp.variables), 
                         constraints=csp.constraints, 
                         solution=csp.solution)
    all_arcs = make_all_arcs(csp)
    agenda = set(all_arcs)
    
    while agenda:
        constraint = agenda.pop()
        if remove_inconsistent_values(constraint, csp):
            X = constraint[0]
            if len(csp.variables[X])==0:
                return False
            neighbors = get_neighbors(constraint, all_arcs)
            agenda.update(neighbors)
    return csp #truthy
    

###############################################################################


# Create a random CSP
csp = make_csp(variables=7, domains=(1,2,3), arcs=0.5,
               return_all_solutions=True, random_seed=True)
draw_graph(csp)

for s,o in zip(csp._fields, csp):
    if s == "solution":
        print("\nsolutions:", len(o))
        continue
    print(f"\n{s}:")
    print(*o, sep='\n')
    if s == "variables":
        print("\ndomains:")
        print(*o.values(), sep='\n')   
print("-"*50, end='\n\n')


#Preprocess with AC3
csp = ac3(csp)
print("domains after ac3:", csp.variables, "\n")


# Solve
assignment = dict()
solution = backtrack(assignment, csp)

print("solution", solution)
print("correct solution?:", solution in csp.solution)

