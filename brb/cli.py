#!/usr/bin/env python
"""Command-line interface for BRB model creation and execution.

While setup is not ready, use it as:
    $ python -m brb.cli rules_file.csv
"""

import click

from .antecedent import *
from .attr_input import AttributeInput
from .brb import csv2BRB


@click.command()
@click.argument('rules', type=click.Path(exists=True))
@click.option('--antecedent-prefix', type=click.STRING, default='A_')
@click.option('--consequent-prefix', type=click.STRING, default='D_')
@click.option('--deltas-prefix', type=click.STRING, default=None)
def main(rules, antecedent_prefix, consequent_prefix, deltas_prefix):
    """Creates a BRB model from rules defined in a csv file.

    The script expects the csv file to have a header, below which each row must
    describe one rule. Each column can be an antecedent or consequent (rule
    weight and attribute weight still not implemented). The prefixes are used to
    identify antecedents and consequents of the model, thus it is crucial that
    they match the columns' names on the csv file.
    """
    _main(rules, antecedent_prefix, consequent_prefix, deltas_prefix)

def _main(rules, antecedent_prefix, consequent_prefix, deltas_prefix):
    model = csv2BRB(
        rules,
        antecedents_prefix=antecedent_prefix,
        consequents_prefix=consequent_prefix,
        deltas_prefix=deltas_prefix
    )
    model = model.expand_rules(matching_method='multiplicative')
    print('Model created')

    # TODO: print instructions

    # get rule input
    print('\nPlease enter the antecedents values (examples between brackets)\n')
    attr_input = dict()
    for U_i in model.U:
        print('Input for {} {}:'.format(U_i.name, U_i))

        antecedent_ref_value = input()

        # we understand that an input is blank for user uncertainty
        if antecedent_ref_value == '':
            if isinstance(U_i, CategoricalAntecedent):
                n_ref_values = len(U_i.referential_values)

                antecedent_ref_value = {
                    ref_value: 1/n_ref_values
                    for ref_value in U_i.referential_values
                }
            elif isinstance(U_i, ContinuousAntecedent):
                antecedent_ref_value = interval[-inf,inf]
            elif isinstance(U_i, DiscreteAntecedent):
                antecedent_ref_value = infset()

        attr_input[U_i.name] = antecedent_ref_value

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
        print('\t{}: {:.2f}'.format(D_j, beta_j))

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
