import numpy as np
from brb.brb import RuleBaseModel, Rule, AttributeInput

if __name__ == "__main__":
    # setup for simple tests
    U = ['Antecedent']
    A = {'Antecedent': ['good', 'bad']}
    D = ['good', 'bad']
    model = RuleBaseModel(U=U, A=A, D=D, F=None)

    model.add_rule(Rule(
        A_values={'Antecedent':'good'},
        beta=[1, 0]  # completely good
    ))

    model.add_rule(Rule(
        A_values={'Antecedent':'bad'},
        beta=[0, 1]  # completely bad
    ))

    # vanishing input
    X = AttributeInput({
        'Antecedent': {
            'good': 0,
            'bad': 0
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [0.0, 0.0]))

    X_1 = AttributeInput({
        'Antecedent': {
            'good': 0.01,
            'bad': 0.01
        }
    })
    X_2 = AttributeInput({
        'Antecedent': {
            'good': 0.0001,
            'bad': 0.0001
        }
    })
    X_3 = AttributeInput({
        'Antecedent': {
            'good': 0.000001,
            'bad': 0.000001
        }
    })
    assert all(
        model.run(X_1) > model.run(X_2)
        and model.run(X_2) > model.run(X_3)
        and [belief_degree >= 0.0 for belief_degree in model.run(X_3)])

    # certain, complete input
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

    # even, complete input
    X = AttributeInput({
        'Antecedent': {
            'good': 0.5,
            'bad': 0.5
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [0.5, 0.5]))

    # uncertain, incomplete, uneven input
    X = AttributeInput({
        'Antecedent': {
            'good': 0.5,
            'bad': 0.0
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [0.5, 0.0]))

    X = AttributeInput({
        'Antecedent': {
            'good': 0.0,
            'bad': 0.5
        }
    })
    belief_degrees = model.run(X)
    assert all(np.isclose(belief_degrees, [0.0, 0.5]))

    # uncertain, uneven input
    X = AttributeInput({
        'Antecedent': {
            'good': 0.3,
            'bad': 0.7
        }
    })
    belief_degrees = model.run(X)

    # matrix input
    model = RuleBaseModel(
        U=['A_1', 'A_2'],
        A={
            'A_1': ['high', 'low'],
            'A_2': ['large', 'medium', 'small']
        },
        D=['RS', 'GP']
    )

    A_ks = np.matrix([
        ['high', 'small'],
        ['high', 'medium'],
        ['high', 'large'],
        ['low', 'small'],
        ['low', 'medium'],
        ['low', 'large']
    ])
    betas = np.matrix(['RS', 'GP', 'GP', 'RS', 'RS', 'GP']).T
    model.add_rules_from_matrix(A_ks=A_ks, betas=betas)

    print('Success!')
