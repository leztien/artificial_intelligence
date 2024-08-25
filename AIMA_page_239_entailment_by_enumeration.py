
"""
a truth table enumeration algorithm that checks entailment
"""


from pandas import DataFrame


# Some helper utilities

def is_atomic(sentence):
    """caters for this particular implementation"""
    return type(sentence) is not tuple


def get_symbols(KB):
    symbols = set()
    def recurse(sentence):
        #nonlocal symbols
        if is_atomic(sentence):
            if sentence is not None:
                symbols.add(sentence)
            return
        op, s1, s2 = sentence
        recurse(s1)
        recurse(s2)
    recurse(KB)
    return sorted(symbols)


# The truth table for the 5 basic operations
truth_table = DataFrame([[1,1,0,0], [0,0,0,1], [0,1,1,1], [1,1,0,1], [1,0,0,1]], 
    dtype=bool,
    index=['NOT', 'AND', 'OR', 'IMP', 'IIF'],
    columns=[(False, False), (False, True), (True, False), (True, True)])



# The meat

def evaluate(sentence, model):
    """
    Evaluates the sentence in the model.
    sentence : a tuple or atomic sentence (str)
    model : dict 
    """
    
    # base case
    if sentence in (False, True):
        return sentence
    if is_atomic(sentence):
        return model[sentence]
    
    # recursive case
    op, s1, s2 = sentence
    s2 = s2 or False  # hackish way to cater for the unary op (NOT)
    
    # Return the value of a particular cell in the Truth-table
    return truth_table.at[op, (evaluate(s1, model), evaluate(s2, model))]


def entails(KB, query):
    """
    True if KB eintails query.
    KB, query : sentences
    """
    # Recursive function that enumerates all models
    def recurse(KB, query, model, symbols):
        # Base case
        if len(symbols) == 0:
            if evaluate(KB, model):
                return evaluate(query, model)
            return True  # if KB is False always return True

        # Recursive case
        symbols = symbols.copy()
        p, *rest = symbols
        
        return recurse(KB, query, {p: False, **model}, rest) \
           and recurse(KB, query, {p: True, **model}, rest) 
    
        
    symbols = get_symbols(KB)
    return recurse(KB, query, dict(), symbols)
    

        

if __name__ == '__main__':

    # Define propositions
    p = "proposition 1"
    q = "proposition 2"
    r = "proposition 3"
    
    # example 1
    KB = ('AND', ('NOT', p, None), ('OR', p, q))
    query = q
    
    ans = entails(KB, query)
    print("KB entails query:", ans)
    
    # example 2
    KB = ('AND', ('IIF', p, q), ('NOT', p, None))
    query = ('NOT', q, None)
    
    ans = entails(KB, query)
    print("KB entails query:", ans)
    

