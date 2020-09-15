import os
import random
import numpy as np
import pandas as pd
from interval import interval, inf
import matplotlib.pyplot as plt
from typing import List, Any
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

def random_existing_input(model, num_runs):

    # create random test inputs using referential values identical with existing ones in the rule base
    res = []
    res_place = []
    counter = 0
    while counter < num_runs:
        counter += 1
        attr_input = dict()

        # generation of a list of the ref_values for each antecedent for random.choice()
        for U_i in model.U:
            ref_vals = []
            for rule in model.rules:
                if U_i in rule.A_values:
                    ref_vals.append(rule.A_values[U_i]) if rule.A_values[U_i] not in ref_vals else ref_vals
            if len(ref_vals) > 0:
                attr_input[U_i] = random.choice(ref_vals)

        # get recommendation for input
        X = AttributeInput(attr_input)
        belief_degrees = model.run(X)
        results = dict(zip(model.D, belief_degrees))

        # ordering starting with the highest values first
        results = {alg: results[alg] for alg in sorted(results, key=results.get, reverse=True)}
        results_place = {alg: num + 1 for num, alg in enumerate(results.keys())}
        print(results_place)
        res.append(results)
        res_place.append(results_place)

    # compute average belief degree over number of runs
    ave_result = {alg: 0 for alg in results.keys()}
    for result in res:
        ave_result = {alg: ave_result[alg] + bel for alg, bel in result.items()}
    ave_result = {alg: value / num_runs for alg, value in ave_result.items()}

    # bringing result data into boxplot plotting format
    boxplot_data = {alg: [] for alg in results.keys()}
    for result in res:
        for alg, bel in result.items():
            boxplot_data[alg].append(bel)
    # sorting
    boxplot_data = {alg: bel for alg, bel in
                    sorted(boxplot_data.items(), key=lambda i: sum(i[1]), reverse=True)}

    boxplot_data_place = {alg: [] for alg in results_place.keys()}
    for result_place in res_place:
        for alg, place in result_place.items():
            boxplot_data_place[alg].append(place)
    # sorting
    boxplot_data_place = {alg: place for alg, place in
                          sorted(boxplot_data_place.items(), key=lambda i: sum(i[1]), reverse=False)}

    # plotting results in boxplot
    title = '{} run(s) on a randomly created, complete input of existing values'.format(num_runs)
    boxplot_results(boxplot_data, title)
    boxplot_results(boxplot_data_place, title)

def custom_input(model, input):
    num_runs = 1
    res = []
    res_place = []
    attr_input = dict()

    # checking how many different inputs there are
    num_inputs = len(input[next(iter(input))])
    for i in range(num_inputs):
        for U_i in model.U:
            attr_input[U_i] = input[U_i][i]
        X = AttributeInput(attr_input)
        belief_degrees = model.run(X)
        results = dict(zip(model.D, belief_degrees))

        # ordering starting with the highest values first
        results = {alg: results[alg] for alg in sorted(results, key=results.get, reverse=True)}
        results_place = {alg: num + 1 for num, alg in enumerate(results.keys())}
        print(results_place)
        res.append(results)
        res_place.append(results_place)

        # compute average belief degree over number of runs
    ave_result = {alg: 0 for alg in results.keys()}
    for result in res:
        ave_result = {alg: ave_result[alg] + bel for alg, bel in result.items()}
    ave_result = {alg: value / num_runs for alg, value in ave_result.items()}

    # bringing result data into boxplot plotting format
    boxplot_data = {alg: [] for alg in results.keys()}
    for result in res:
        for alg, bel in result.items():
            boxplot_data[alg].append(bel)
    # sorting
    boxplot_data = {alg: bel for alg, bel in
                    sorted(boxplot_data.items(), key=lambda i: sum(i[1]), reverse=True)}

    boxplot_data_place = {alg: [] for alg in results_place.keys()}
    for result_place in res_place:
        for alg, place in result_place.items():
            boxplot_data_place[alg].append(place)
    # sorting
    boxplot_data_place = {alg: place for alg, place in
                          sorted(boxplot_data_place.items(), key=lambda i: sum(i[1]), reverse=False)}

    # plotting results in boxplot
    title = 'Custom input'
    boxplot_results(boxplot_data, title)
    boxplot_results(boxplot_data_place, title)


def boxplot_results(data: List[any], title):
    _data = [np.asarray(results) for results in data.values()]
    _consequents = [key for key in data.keys()]

    plt.boxplot(_data, labels=_consequents)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# inputs for klein2019 custom input
inputs_klein = {
          "Dimensionality of HPs": [8, 8, 8, 8],
          "Conditional HP Space": ['no', 'no', 'no', 'no'],
          "#continuous HPs of ML alg.": ['>=1', '0', '>=1', '0'],
          "Number of possible function evaluations/maximum number of trials": ['<100', '<100', '<100', '<100'],
          "Machine Learning Algorithm": ['SVM', 'XGBoost', 'XGBoost', 'XGBoost'],
          "Dataset to perform ML task": ['10 UCI Regression datasets', '16 OpenML classification datasets', '16 OpenML classification datasets', '16 OpenML classification datasets'],
          "Artificial noise in dataset": ['no', 'no', 'no', 'yes'],
          "Surrogate Benchmarking": ['yes', 'yes', 'yes', 'yes'],
          "Task that was performed by the ML algorithm who's HPs were optimized": ['Regression', 'Classification', 'Classification', 'Classification']}

# inputs BeliefRuleBase_v3
inputs_BRB_v3 = {
    'A_UR: quality demands':
        ['low'],
    'A_UR: computational efficiency of the HPO technique':
        [''],
    'A_User\'s programming ability':
        [''],
    'A_Access to parallel computing':
        [''],
    'A_Production use case':
        [''],
    'A_Time Resources':
        [''],
    'A_Number of maximum function evaluations/ trials budget':
        [''],
    'A_Cummulative Budget':
        [''],
    'A_Wall Clock Time [s]':
        [''],
    'A_Running time per trial [s]':
        [''],
    'A_Machine Learning Algorithm':
        [''],
    'A_Obtainability of good approximate':
        [''],
    'A_Supports parallel evaluations':
        [''],
    'A_Usage of one-hot encoding for cat. features':
        [''],
    'A_Dimensionality of HPs':
        [''],
    'A_Conditional HP space':
        [''],
    'A_#continuous HPs of ML alg.':
        [''],
    'A_Obtainability of gradients':
        [''],
    'A_Dataset (name)':
        [''],
    'A_Number of instances in dataset':
        [''],
    'A_Artificial noise in dataset':
        [''],
    'A_Surrogate benchmarking':
        [''],
    'A_Validation':
        [''],
    'A_ML task':
        ['Regression']
}
curdir_path = '/Users/philippnoodt/VirtualBox_VMs/Win10/Win10_SharedFolder/MA/coding/Bruno/git/brb/'

if __name__ == "__main__":

    # create model from rules.csv
    model = csv2BRB('rulebase_v3.csv', antecedents_prefix='A_', consequents_prefix='D_')
    print('Model created')

    # test with random, existing inputs
    #random_existing_input(model, 10)

    # test with custom inputs
    custom_input(model, inputs_BRB_v3)
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