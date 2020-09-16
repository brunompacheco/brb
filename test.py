import os
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from interval import interval, inf

from brb.attr_input import AttributeInput, str2interval
from brb.brb import RuleBaseModel, csv2BRB
from brb.rule import Rule

if __name__ == "__main__":
    # setup for simple tests
    U = ['Antecedent']
    D = ['good', 'bad']
    model = RuleBaseModel(U=U, D=D, F=None)

    good_rule = Rule(
        A_values={'Antecedent':'good'},
        beta=[1, 0]  # completely good
    )
    bad_rule = Rule(
        A_values={'Antecedent':'bad'},
        beta=[0, 1]  # completely bad
    )
    model.add_rule(good_rule)
    model.add_rule(bad_rule)

    # Arithmetic matching degree calculation
    good_rule.matching_degree = 'arithmetic'
    X = AttributeInput({
        'Antecedent': {
            'good': 0,
            'bad': 0
        }
    })
    assert good_rule.get_matching_degree(X) == 0.0
    X = AttributeInput({
        'Antecedent': {
            'good': 0,
            'bad': 0
        }
    })
    assert good_rule.get_matching_degree(X) == 0.0

    # Matching degrees boundaries
    def obj_function(A, rule):
        X = AttributeInput({
            'Antecedent': {
                'good': A[0],
                'bad': A[1]
            }
        })
        return rule.get_matching_degree(X)

    res = minimize(obj_function, [1,1], args=good_rule, bounds=[(0,1), (0,1)])
    assert res['success'] == True
    assert res['fun'] >= 0

    res = minimize(obj_function, [1,1], args=bad_rule, bounds=[(0,1), (0,1)])
    assert res['success'] == True
    assert res['fun'] >= 0

    def obj_function(A, rule):
        X = AttributeInput({
            'Antecedent': {
                'good': A[0],
                'bad': A[1]
            }
        })
        return - rule.get_matching_degree(X)

    res = minimize(obj_function, [1,1], args=good_rule, bounds=[(0,1), (0,1)])
    assert res['success']
    assert - res['fun'] <= 1

    res = minimize(obj_function, [1,1], args=bad_rule, bounds=[(0,1), (0,1)])
    assert res['success']
    assert - res['fun'] <= 1

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
    betas = np.matrix([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    model.add_rules_from_matrix(A_ks=A_ks, betas=betas)

    assert len(model.rules) == len(A_ks)
    for A_k, rule in zip(A_ks, model.rules):
        assert (A_k == list(rule.A_values.values())).all()

    # interval string check
    not_intervals = ['', 'word', '12', '1.2', '<3]', '2:']
    for not_interval in not_intervals:
        try:
            _ = str2interval(not_interval)
            raise AssertionError(
                not_interval + ' should not be converted to interval'
            )
        except:
            pass

    true_intervals = ['1:2', '1.0: 2', '1.2: 2.1', ' 1:  2 ']
    for true_interval in true_intervals:
        _ = str2interval(true_interval)

    # antecedent matching degree
    antecedents_matchings = [
        # easy referential values
        (['A', 'A'], 1.0),
        (['A', 'B'], 0.0),
        ([12, 12], 1.0),
        ([12, 9], 0.0),
        # uncertain input
        (['A', {'A': 0.7, 'B': 0.3}], 0.7),
        # interval referential values
        ([{1,2}, {1,2}], 1.0),
        ([{1,2}, {2,3}], 0.5),
        ([{1,2}, {3,4}], 0.0),
        ([interval[1,2], interval[1,2]], 1.0),
        ([interval[1,2], interval[2,3]], 0.0),
        ([interval[1,2], interval[3,4]], 0.0),
        ([interval[1,2], interval[1.5,2.5]], 0.5),
        ([interval[1.5,2], interval[1.5,2.5]], 0.5),
        ([interval[1,2], interval[1.5,2.0]], 1.0),
        # infinite intervals
        ([interval[1,inf], interval[1.5,50]], 1.0),
        ([interval[1,inf], interval[0,2]], 0.5),
        # mixed
        ([{1,2}, 2], 1.0),
        ([{1,2}, 3], 0.0),
        ([interval[1,2], 2], 1.0),
        ([interval[1,2], 3], 0.0),
        ([interval[2.5, 4.5], {1,2,3,4}], 0.5),
        ([interval[0.5, 4.5], {1,2,3,4}], 1.0),
        ([interval[2.5, 4.5], {1,2}], 0.0),
    ]
    for antecedents, expected_match in antecedents_matchings:
        assert Rule._get_antecedent_matching(antecedents[1], antecedents[0]) == expected_match

    # referential value preparation
    referential_values = [
        ('word', str),
        ('12', int),
        (12, int),
        ('1.27', float),
        (1.27, float),
        ('1:3', set),
        ({1, 2, 3}, set),
        ('1.0: 3.0', interval),
        (' 1.0  : 3.0 ', interval),
        (interval[1,3], interval),
        ("{'A':0.6, 'B': 0.4}", dict),
        ({'A':0.6, 'B': 0.4}, dict),
    ]
    for referential_value, refv_type in referential_values:
        assert isinstance(AttributeInput.prep_referential_value(referential_value), refv_type)

    # interval-based rules
    model = RuleBaseModel(
        U=['A_1', 'A_2'],
        D=['Y', 'N']
    )

    # numerical input, interval rule
    rule = Rule(
        A_values={'A_1':'1:2', 'A_2':'1.0:2.0'},
        beta=[1, 0]
    )
    input_matches = [
        (AttributeInput({'A_1': 1, 'A_2': '0.999'}), 0.5),
        (AttributeInput({'A_1': '1', 'A_2': '1.999'}), 1.0),
        (AttributeInput({'A_1': 0, 'A_2': '1: 1.5'}), 0.5),
    ]
    for X, expected_matching_degree in input_matches:
        assert rule.get_matching_degree(X) == expected_matching_degree

    # interval input, numerical rule
    rule = Rule(
        A_values={'A_1':'3', 'A_2':3},
        beta=[0, 1]
    )
    input_matches = [
        (AttributeInput({'A_1': 3, 'A_2': '3'}), 1.0),
        (AttributeInput({'A_1': '2:3', 'A_2': '1.999'}), 0.25),
        (AttributeInput({'A_1': '3', 'A_2': '1: 3.5'}), 1.0),
    ]
    for X, expected_matching_degree in input_matches:
        assert rule.get_matching_degree(X) == expected_matching_degree

    # dataframe rule input
    rules_filepath = os.path.join(os.curdir, 'test_rules.csv')
    df_rules = pd.read_csv(rules_filepath, index_col='rule_id')

    df_cols = df_rules.columns

    U = [col for col in df_cols if col[:2] == 'A_']
    D = [col for col in df_cols if col[:2] == 'D_']

    model = RuleBaseModel(
        U=U,
        D=D
    )

    model.add_rules_from_df(rules_df=df_rules)

    assert len(model.rules) == df_rules.shape[0]

    input_matches = [
        (AttributeInput(dict(zip(U, ['Yes', '1:2', 1.5]))), 1.0),
        (AttributeInput(dict(zip(U, [{'Yes':0.5, 'No':0.5}, '0:1', '0.5:1.5']))), 0.5),
    ]
    for X, expected_matching_degree in input_matches:
        assert model.rules[-2].get_matching_degree(X) == expected_matching_degree

    input_matches = [
        (AttributeInput(dict(zip(U, ['No', '1.0:2.0', 3.5]))), 1.0),
        (AttributeInput(dict(zip(U, [{'Yes':0.5, 'No':0.5}, '0:2.0', '2.2:4.2']))), 0.5),
    ]
    for X, expected_matching_degree in input_matches:
        assert model.rules[-1].get_matching_degree(X) == expected_matching_degree

    # csv rule input
    rules_filepath = os.path.join(os.curdir, 'test_rules.csv')

    csv_model = csv2BRB(rules_filepath, antecedents_prefix='A_', consequents_prefix='D_')

    assert len(csv_model.rules) == len(model.rules)

    # AttributeInput
    expected_ref_value_conversions = [
        (12, int),
        ('12', int),
        (12.0, float),
        ('12.0', float),
        ('asdf', str),
        ('1:2', set),
        ('1.0:2', interval),
        ('1.0:2.0', interval),
        ('>1', interval),
        ('<1', interval),
        ({'A': 0.5, 'B':0.5}, dict),
        ("{'A': 0.5, 'B':0.5}", dict),
    ]
    for X_i, expected_type in expected_ref_value_conversions:
        assert isinstance(AttributeInput.prep_referential_value(X_i), expected_type)

    X = AttributeInput({
        'A_1': 12,
        'A_2': {'y': 0.5, 'n':0.0},
        'A_3': '1:2',
    })
    assert X.get_completeness(['A_1', 'A_3']) == 1.0
    assert X.get_completeness(['A_2']) == 0.5
    assert X.get_completeness(['A_2', 'A_3']) == 0.75

    print('Success!')
