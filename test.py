from brb.brb import *

if __name__ == "__main__":
    U = ['Antecedant']
    A = {'Antecedant': ['good', 'bad']}
    D = ['bad', 'good']
    model = RuleBaseModel(U=U, A=A, D=D, F=None)

    model.add_rule(Rule(
        A_values={'Antecedant':'good'},
        delta={'Antecedant':1},
        theta=1,
        beta={'good':1, 'bad':0}
    ))

    model.add_rule(Rule(
        A_values={'Antecedant':'bad'},
        delta={'Antecedant':1},
        theta=1,
        beta={'good':0, 'bad':1}
    ))

    X = AttributeInput({
        'Antecedant': [1.0, 0]
    })
    belief_degrees, remaining_belief_degree = model.run(X)

    X = AttributeInput({
        'Antecedant': [0, 1.0]
    })
    belief_degrees, remaining_belief_degree = model.run(X)
    pass

    #U = ['color', 'speed']
    #A = {'color':['red','green','blue'],'speed':[0,5,10]}
    #D = ['good', 'bad']

    #model = RuleBaseModel(U=U, A=A, D=D, F=None)

    #A_values = {
        #'color': 'red',
        #'speed': 10
    #}
    #delta = {'color': 1, 'speed': 1}
    #r = Rule(A_values=A_values, delta=delta, theta=1, beta={'good':0,'bad':1})