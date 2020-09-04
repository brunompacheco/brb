#!/usr/bin/env python
"""Command-line interface for BRB model creation and execution.

While setup is not ready, use it as:
    $ python -m brb.cli rules_file.csv
"""

import click

from interval import interval, inf

from .brb import csv2BRB
from .attr_input import AttributeInput


@click.command()
@click.argument('rules', type=click.Path(exists=True))
@click.option('--antecedent-prefix', type=click.STRING, default='A_')
@click.option('--consequent-prefix', type=click.STRING, default='D_')
def main(rules, antecedent_prefix, consequent_prefix):
    """Creates a BRB model from rules defined in a csv file.

    The script expects the csv file to have a header, below which each row must
    describe one rule. Each column can be an antecedent or consequent (rule
    weight and attribute weight still not implemented). The prefixes are used to
    identify antecedents and consequents of the model, thus it is crucial that
    they match the columns' names on the csv file.
    """
    _main(rules, antecedent_prefix, consequent_prefix)

def _main(rules, antecedent_prefix, consequent_prefix):
    model = csv2BRB(
        rules,
        antecedents_prefix=antecedent_prefix,
        consequents_prefix=consequent_prefix
    )
    print('Model created')

    # TODO: print instructions

    # get attributes possible input
    attr_values = dict()
    for U_i in model.U:
        attr_values[U_i] = set()
        for rule in model.rules:
            A_i = rule.A_values[U_i]

            # format string version
            if isinstance(A_i, interval):
                A_i = A_i[0]  # only first component is considered, always
                if A_i[0] == -inf:
                    A_i = '<{}'.format(A_i[-1])
                elif A_i[1] == inf:
                    A_i = '>{}'.format(A_i[0])
                else:
                    A_i = '{}:{}'.format(*A_i)
            elif isinstance(A_i, set):
                A_i = '{}:{}'.format(min(A_i), max(A_i))
            else:
                A_i = str(A_i)

            assert isinstance(A_i, str)

            attr_values[U_i].add(A_i)

    # get rule input
    print('\nPlease enter the antecedents values (examples between brackets)\n')
    attr_input = dict()
    for U_i, A_i in attr_values.items():
        print('Input for {} {}:'.format(U_i, A_i))

        attr_input[U_i] = input()

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
    main()  # pylint: disable=no-value-for-parameter
