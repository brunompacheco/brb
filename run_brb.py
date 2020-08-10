import click

import numpy as np
import pandas as pd

from brb.brb import RuleBaseModel, AttributeInput


def get_model_from_csv(csv_path: str) -> RuleBaseModel:
    # import rules
    rules = pd.read_csv(csv_path, sep=';', index_col=0)
    rules = rules[rules.columns[1:]]  # drop weight column (empty)

    # split attributes
    consequent = rules.iloc[:, -1]
    antecedents = rules.iloc[:, :-1]

    # model setup
    U = antecedents.columns
    D = consequent.unique()
    A = {U_i: antecedents[U_i].dropna().unique() for U_i in U}
    model = RuleBaseModel(U=U, A=A, D=D)

    # add rules
    A_ks = np.matrix(antecedents.values)
    model.add_rules_from_matrix(
        A_ks=A_ks,
        betas=consequent.values
    )

    return model

@click.command()
@click.argument('rules', type=click.Path(exists=True))
def _main(rules):
    model = get_model_from_csv(rules)
    print('Model created')

    # get rule input
    print('\nPlease enter the antecedents values in the order of the referential values displayed, separated by commas\n')
    attr_input = dict()
    for U_i, A_i in model.A.items():
        print('Input for {} {}:'.format(U_i, A_i))
        user_input = input()
        user_inputs = user_input.replace(' ', '').split(',')

        if user_inputs[0] == '':
            user_inputs[0] = '0'

        while len(user_inputs) < len(A_i):
            user_inputs.append('0')

        user_inputs = [float(user_input) for user_input in user_inputs]

        attr_input[U_i] = dict(zip(A_i, user_inputs))

    X = AttributeInput(attr_input)
    belief_degrees = model.run(X)

    # Display rules and its activations with the results
    print('\nActivated Rules:')

    matching_degrees = [rule.get_matching_degree(X) for rule in model.rules]

    for rule, matching_degree in zip(model.rules, matching_degrees):
        if matching_degree > 0:
            print("[Matching Degree: {}] {}".format(matching_degree, rule))

    print('\nResult:')
    for D_j, beta_j in zip(model.D, belief_degrees):
        print('\t{}: {}'.format(D_j, beta_j))

if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
