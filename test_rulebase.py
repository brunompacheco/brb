import pandas as pd
import numpy as np

from interval import interval, inf

from brb.brb import csv2BRB
from brb.attr_input import AttributeInput


model = csv2BRB('rulebase_v2.csv', antecedents_prefix='A_', consequents_prefix='D_'
)
print('Model created')

# TODO: print instructions

# get attributes possible input
attr_values = dict()
for U_i in model.U:
    attr_values[U_i] = set()
    for rule in model.rules:
        if U_i in rule.A_values.keys():
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