from brb.brb import *

if __name__ == "__main__":
    U = ['Antecedent']
    A = {'Antecedent': ['good', 'bad']}
    D = ['bad', 'good']
    model = RuleBaseModel(U=U, A=A, D=D, F=None)

    model.add_rule(Rule(
        A_values={'Antecedent':'good'},
        delta={'Antecedent':1},
        theta=1,
        beta=[1, 0]  # completely good
    ))

    model.add_rule(Rule(
        A_values={'Antecedent':'bad'},
        delta={'Antecedent':1},
        theta=1,
        beta=[0, 1]  # completely bad
    ))

    X = AttributeInput({
        'Antecedent': {
            'good': 1.0,
            'bad': 0.0
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [1.0, 0.0]))

    X = AttributeInput({
        'Antecedent': {
            'good': 0.0,
            'bad': 1.0
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [0.0, 1.0]))

    X = AttributeInput({
        'Antecedent': {
            'good': 0.5,
            'bad': 0.5
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [0.5, 0.5]))

    X = AttributeInput({
        'Antecedent': {
            'good': 0.5,
            'bad': 0.0
        }
    })
    belief_degrees = model.run(X)
    # assert all(np.isclose(belief_degrees, [0.5, 0.0]))

    X = AttributeInput({
        'Antecedent': {
            'good': 0.0,
            'bad': 0.5
        }
    })
    belief_degrees = model.run(X)
    # assert all(np.isclose(belief_degrees, [0.0, 0.5]))

    X = AttributeInput({
        'Antecedent': {
            'good': 0.3,
            'bad': 0.7
        }
    })
    belief_degrees = model.run(X)

    print('Success!')
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