import os
import random
import numpy as np
import pandas as pd
from interval import interval, inf

from brb.brb import csv2BRB
from brb.attr_input import AttributeInput


def enter_custom_input(A_i, X_i):
    user_inputs = A_i.copy()
    for idx, ref_val in enumerate(A_i):
        if ref_val == X_i:
            user_inputs[idx] = 1
        else:
            user_inputs[idx] = 0
    return user_inputs

def custom_input(input):
    attr_input = dict()
    num = len(input[next(iter(input))])
    for i in range(num):
        for U_i, A_i in model.A.items():
            user_inputs = enter_custom_input(A_i, input[U_i][i])
            attr_input[U_i] = dict(zip(A_i, user_inputs))
        X = AttributeInput(attr_input)
        belief_degrees = model.run(X)
        print(attr_input)
        print(tuple(zip(model.D, belief_degrees)))

# inputs for klein2019 custom input
inputs = {"Dimensionality of HPs": [8, 8, 8, 8],
          "Conditional HP Space": ['no', 'no', 'no', 'no'],
          "#continuous HPs of ML alg.": ['>=1', '0', '>=1', '0'],
          "Number of possible function evaluations/maximum number of trials": ['<100', '<100', '<100', '<100'],
          "Machine Learning Algorithm": ['SVM', 'XGBoost', 'XGBoost', 'XGBoost'],
          "Dataset to perform ML task": ['10 UCI Regression datasets', '16 OpenML classification datasets', '16 OpenML classification datasets', '16 OpenML classification datasets'],
          "Artificial noise in dataset": ['no', 'no', 'no', 'yes'],
          "Surrogate Benchmarking": ['yes', 'yes', 'yes', 'yes'],
          "Task that was performed by the ML algorithm who's HPs were optimized": ['Regression', 'Classification', 'Classification', 'Classification']}

curdir_path = '/Users/philippnoodt/VirtualBox_VMs/Win10/Win10_SharedFolder/MA/coding/Bruno/git/brb/'

if __name__ == "__main__":

    # create model from rules.csv
    model = csv2BRB('rulebase.csv', antecedents_prefix='A_', consequents_prefix='D_')
    print('Model created')

    # create random test inputs using referential values identical with existing ones in the rule base

    num_tests = 10
    counter = 0
    while counter < num_tests:
        counter += 1

        attr_input = dict()
        for U_i in model.U:
            ref_vals = []
            for rule in model.rules:
                if U_i in rule.A_values:
                    ref_vals.append(rule.A_values[U_i]) if rule.A_values[U_i] not in ref_vals else ref_vals
            if len(ref_vals) > 0:
                attr_input[U_i] = random.choice(ref_vals)
            '''
            rand_ref_val = random.choice(A_i)
            user_inputs = A_i.copy()
            for idx, ref_val in enumerate(A_i):
                if ref_val == rand_ref_val:
                    user_inputs[idx] = 1
                else:
                    user_inputs[idx] = 0
            attr_input[U_i] = dict(zip(A_i, user_inputs))
            '''

        X = AttributeInput(attr_input)
        belief_degrees, combined_belief_degrees = model.run(X)
        #print(attr_input)
        print(tuple(zip(model.D, belief_degrees)))
        print(tuple(zip(model.D, combined_belief_degrees)))

    '''
    # create random test inputs using new referential values
    print('\nindividual test inputs')
    custom_input(inputs)
    '''

    print('success')

'''

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

'''