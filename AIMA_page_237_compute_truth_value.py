
"""
Compute the truth value of a proporitionla logi sentence
"""


from pandas import DataFrame


# Construct the truth table
truth_table = DataFrame([[1,1,0,0], [0,0,0,1], [0,1,1,1], [1,1,0,1], [1,0,0,1]], 
    dtype=bool,
    index=['NOT', 'AND', 'OR', 'IMP', 'IIF'],
    columns=[(False, False), (False, True), (True, False), (True, True)])


def evaluate(sentence, model):
    """
    Evaluates the sentence in the model.
    sentence : a tuple or atomic sentence (str)
    model : dict 
    """
    def is_atomic(sentence):
        """caters for this particular implementation"""
        return type(sentence) is not tuple
    
    # base case
    if sentence in (False, True):
        return sentence
    if is_atomic(sentence):
        return model[sentence]
    
    # recursive case
    op, s1, s2 = sentence
    s2 = s2 or False  # hackish way to cater for the unary op (NOT)
    return truth_table.at[op, (evaluate(s1, model), evaluate(s2, model))]

    

if __name__ == '__main__':

    # Define propositions
    p = "proposition 1"
    q = "proposition 2"
    r = "proposition 3"
    
    # Constuct a sentence
    s = ('AND', p, q)
    s = ('NOT', s, None)
    s = ('OR', s, r)
    s = ('IMP', r, s)
    s = ('IIF', s, ('OR', p, q))
    
    # Define a model
    m = {p: False, q: True, r: False}
    
    b = evaluate(sentence=s, model=m)
    print("ANS:", b)


