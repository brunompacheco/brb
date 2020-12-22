import os
import random
import numpy as np
import pandas as pd
from interval import interval, inf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import seaborn as sns
from typing import List, Any
from brb.brb import csv2BRB
from brb.attr_input import AttributeInput
import math

mpl.rcParams['font.family'] = 'Arial'
# 179c7d is IPTgreen
palette = sns.light_palette("#179c7d", as_cmap=True)
matplotlib.cm.register_cmap("IPTgreencmap", palette)
sns.set_theme(style="whitegrid")

def enter_custom_input(A_i, X_i):
    user_inputs = A_i.copy()
    for idx, ref_val in enumerate(A_i):
        if ref_val == X_i:
            user_inputs[idx] = 1
        else:
            user_inputs[idx] = 0
    return user_inputs

def random_existing_input(model, num_runs, incomplete, rec):
    """Creates a random input using the referential
    values that are exisiting in the rule base.

    Args:
        model: model to create input from.
        num_runs: number of runs to evaluate.
        incomplete: takes bool or float between 0 and 1.
            if bool, '' is added to the list of referential
            values that is randomly chosen from. if float,
            this is the probability that the input for a certain
            antecedent is empty.
    """

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

                # enables random incomplete input
                if incomplete == True:
                    ref_vals.append('')
                    attr_input[U_i.name] = random.choice(ref_vals)
                elif isinstance(incomplete, float):
                    random_val = random.random()
                    if incomplete > random_val:
                        attr_input[U_i.name] = ''
                    else:
                        attr_input[U_i.name] = random.choice(ref_vals)
                else:
                    attr_input[U_i.name] = random.choice(ref_vals)

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
    if incomplete == 'True':
        complete = 'incomplete'
    else:
        complete = int(incomplete*100)

    title = '{} runs on a randomly created input of existing values. User specified {}% of the antecedents.'.format(num_runs, complete)
    boxplot_results(boxplot_data, title, y='Final belief', rec=rec)
    boxplot_results(boxplot_data_place, title, y='Average rank', rec=rec)

def custom_input(model, input, rec, show_top):
    num_runs = 1
    res = []
    res_place = []
    attr_input = dict()

    # checking how many different inputs there are
    num_inputs = len(input[next(iter(input))])
    for i in range(num_inputs):
        for U_i in model.U:
            if len(input[U_i.name]) > 1:
                attr_input[U_i.name] = input[U_i.name][i]
            else:
                attr_input[U_i.name] = input[U_i.name]
        X = AttributeInput(attr_input)
        belief_degrees = model.run(X)
        results = dict(zip(model.D, belief_degrees))

        # ordering starting with the highest values first
        results = {alg: results[alg] for alg in sorted(results, key=results.get, reverse=True)}
        print(results)
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
    boxplot_custominputs_results(res, title, 'Total belief', rec=rec, show_top=show_top)
    #boxplot_results(boxplot_data, title)
    #boxplot_results(boxplot_data_place, title)

def boxplot_results(data: List[any], title, y, rec):
    _data = [np.asarray(results) for results in data.values()]
    _consequents = [key.split('_')[1] for key in data.keys()]

    _dict = {y: [], rec: []}
    for key in data.keys():
        for value in data[key]:
            _dict[y].append(value)
            _dict[rec].append(key.split('_')[1])
    _df = pd.DataFrame.from_dict(_dict)

    #plt.boxplot(_data, labels=_consequents)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')

    sns.boxplot(x=rec, y=y, data=_df, palette='IPTgreencmap')
    plt.tight_layout()

    plt.show()

def boxplot_custominputs_results(data: List[any], title, y, rec, show_top):
    sqrt = math.ceil(np.sqrt(len(data)))
    fig, axes = plt.subplots(sqrt, sqrt)

    title_HPO_KO1 = ['IF {(Transparency; yes)}',
                  'IF {(Transparency; must)}',
                  'IF {(Well-documented implementation; must)}',
                  'IF {(Conditionality; yes)}'
                  ]
    title_HPO_KO4 = ['IF {(Transparency; must)}\nIF {(Well-documented implementation; must)}\nIF {(Conditionality; yes)}',
                     'IF {(Transparency; must)}\nIF {(Well-documented implementation; must)}',
                     'IF {(Transparency; must)}\nIF {(Conditionality; yes)}',
                     'IF {(Conditionality; yes)}'
                     ]
    titles_ML_3UCs = ['Use case 1: Learning ML beginner',
                      'Use case 2: Proof-of-concept',
                      'Use case 3: High performance',
                      '-'
                      ]

    for idx, result in enumerate(data):
        _dict = {y: [], rec: []}
        _data = [np.asarray(result) for result in result.values()]
        _consequents = [key.split('_')[1] for key in result.keys()]

        for key in result.keys():
            _dict[y].append(result[key])
            _dict[rec].append(key.split('_')[1])
        _df = pd.DataFrame.from_dict(_dict)
        if show_top == 'all':
            pass
        else:
            _df = _df[:show_top]
            _consequents = _consequents[:show_top]
        #axes[math.floor(idx/sqrt), idx % sqrt].boxplot(_data)
        #axes[math.floor(idx/sqrt), idx % sqrt].set_xticklabels(labels=_consequents, rotation=45, ha='right')
        sns.boxplot(ax=axes[math.floor(idx / sqrt), idx % sqrt], x=rec, y=y, data=_df,
                    palette='IPTgreencmap')
        y_title_margin = 1.2
        #sns.set_theme(font="Arial", font_scale=6)
        axes[math.floor(idx/sqrt), idx % sqrt].set_title(title_HPO_KO4[idx], fontsize=9) #, y=y_title_margin
        axes[math.floor(idx/sqrt), idx % sqrt].set_xticklabels(labels=_consequents, rotation=45,
                                                               ha='right', fontsize=7) #
        #axes[math.floor(idx / sqrt), idx % sqrt].set_yticklabels(plt.yticks(), fontsize=7)
        axes[math.floor(idx / sqrt), idx % sqrt].set_xlabel('')
        axes[math.floor(idx / sqrt), idx % sqrt].set_ylabel('Total belief', fontsize=9)
        #sns.set_context("paper", rc={"font.size": 7, "axes.titlesize": 9, "axes.labelsize": 7})

    fig.subplots_adjust(top=0.95, bottom=0.16, left=0.12, right=0.95, hspace=0.72, wspace=0.4)
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

# inputs HPO BeliefRuleBase_v9 - Bruno's three cases
inputs_HPO_BRB_v9_3cases = {
    'A_UR: quality demands':
        ['', '', 'high', 'high'],
    'A_User\'s programming ability':
        ['low', 'low', 'high', 'high'],
    'A_UR: need for model transparency':
        ['yes', '', '', 'yes'],
    'A_UR: Availability of a well documented library':
        ['yes', '', '', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Access to parallel computing':
        ['', '', 'yes', 'yes'],
    'A_Production use case':
        ['', '', '', ''],  #Predictive Quality
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Number of kernels used':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['>172800', '<7200', '>172800', '>172800'],
    'A_Machine Learning Algorithm':
        ['', '', '', ''],
    'A_Obtainability of good approximate':
        ['', '', '', ''],
    'A_Supports parallel evaluations':
        ['', '', '', ''],
    'A_Usage of one-hot encoding for cat. features':
        ['', '', '', ''],
    'A_Dimensionality of HPs':
        ['', '', '', ''],
    'A_Conditional HP space':
        ['', '', '', ''],
    'A_HP datatypes':
        ['', '', '', ''],
    'A_Availability of a warm-start HP configuration':
        ['', '', '', ''],
    'A_Obtainability of gradients':
        ['', '', '', ''],
    'A_Input Data':
        ['', '', '', ''],   #Image data, Tabular data
    'A_#Instances training dataset':
        ['', '', '', ''],
    'A_Ratio training to test dataset':
        ['', '', '', ''],
    'A_Dataset balance':
        ['', '', '', ''],  #imbalanced
    'A_Ratio positive to negative targets':
        ['', '', '', ''],
    'A_Noise in dataset':
        ['', '', '', ''],
    'A_Training Technique':
        ['', '', '', ''],
    'A_ML task':
        ['', '', '', ''],   #Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   #Image Recognition
}

# inputs HPO BeliefRuleBase_v13 - KNOCK-OUT RULES TESTING 1
inputs_HPO_BRB_KO1_v13 = {
    'A_UR: quality demands':
        ['', '', '', ''],
    'A_User\'s programming ability':
        ['', '', '', ''],
    'A_UR: need for model transparency':
        ['yes', 'must', '', ''],
    'A_UR: Availability of a well documented library':
        ['', '', 'must', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: Number of workers/kernels for parallel computing':
        ['', '', '', ''],
    'A_Production application area':
        ['', '', '', ''],  # 'Predictive Quality'
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['', '', '', ''],  # >172800, '7200.0:172800'
    'A_Machine Learning Algorithm':
        ['', '', '', ''],
    'A_Obtainability of good approximate':
        ['', '', '', ''],
    'A_Supports parallel evaluations':
        ['', '', '', ''],
    'A_Dimensionality of HPs':
        ['', '', '', ''],
    'A_Conditional HP space':
        ['', '', '', 'yes'],
    'A_HP datatypes':
        ['', '', '', ''],
    'A_Availability of a warm-start HP configuration':
        ['', '', '', ''],
    'A_Obtainability of gradients':
        ['', '', '', ''],
    'A_Input Data':
        ['', '', '', ''],  # Image data
    'A_#Instances training dataset':
        ['', '', '', ''],
    'A_Ratio training to test dataset':
        ['', '', '', ''],
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training Technique':
        ['', '', '', ''],   # offline
    'A_ML task':
        ['', '', '', ''],   # Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}
# inputs HPO BeliefRuleBase_v13 - KNOCK-OUT RULES TESTING 2
inputs_HPO_BRB_KO2_v13 = {
    'A_UR: quality demands':
        ['high', 'high', 'high', 'high'],
    'A_User\'s programming ability':
        ['', '', '', ''],
    'A_UR: need for model transparency':
        ['yes', 'must', '', ''],
    'A_UR: Availability of a well documented library':
        ['', '', 'must', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: Number of workers/kernels for parallel computing':
        ['', '', '', ''],
    'A_Production application area':
        ['', '', '', ''],  # 'Predictive Quality'
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['7200.0:172800', '7200.0:172800', '7200.0:172800', '7200.0:172800'],  # >172800, '7200.0:172800'
    'A_Machine Learning Algorithm':
        ['XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
    'A_Obtainability of good approximate':
        ['', '', '', ''],
    'A_Supports parallel evaluations':
        ['', '', '', ''],
    'A_Dimensionality of HPs':
        ['', '', '', ''],
    'A_Conditional HP space':
        ['', '', '', 'yes'],
    'A_HP datatypes':
        ['', '', '', ''],
    'A_Availability of a warm-start HP configuration':
        ['', '', '', ''],
    'A_Obtainability of gradients':
        ['', '', '', ''],
    'A_Input Data':
        ['', '', '', ''],  # Image data
    'A_#Instances training dataset':
        ['', '', '', ''],
    'A_Ratio training to test dataset':
        ['', '', '', ''],
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training Technique':
        ['Offline', 'Offline', 'Offline', 'Offline'],   # Offline
    'A_ML task':
        ['Multiclass Classification', 'Multiclass Classification', 'Multiclass Classification', 'Multiclass Classification'],   # Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}
# inputs HPO BeliefRuleBase_v13 - KNOCK-OUT RULES TESTING 3
inputs_HPO_BRB_KO3_v13 = {
    'A_UR: quality demands':
        ['high', 'high', 'high', 'high'],
    'A_User\'s programming ability':
        ['', '', '', ''],
    'A_UR: need for model transparency':
        ['must', 'must', 'must', ''],
    'A_UR: Availability of a well documented library':
        ['must', 'must', '', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: Number of workers/kernels for parallel computing':
        ['', '', '', ''],
    'A_Production application area':
        ['', '', '', ''],  # 'Predictive Quality'
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['7200.0:172800', '7200.0:172800', '7200.0:172800', '7200.0:172800'],  # >172800, '7200.0:172800'
    'A_Machine Learning Algorithm':
        ['XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
    'A_Obtainability of good approximate':
        ['', '', '', ''],
    'A_Supports parallel evaluations':
        ['', '', '', ''],
    'A_Dimensionality of HPs':
        ['', '', '', ''],
    'A_Conditional HP space':
        ['yes', '', 'yes', 'yes'],
    'A_HP datatypes':
        ['', '', '', ''],
    'A_Availability of a warm-start HP configuration':
        ['', '', '', ''],
    'A_Obtainability of gradients':
        ['', '', '', ''],
    'A_Input Data':
        ['', '', '', ''],  # Image data
    'A_#Instances training dataset':
        ['', '', '', ''],
    'A_Ratio training to test dataset':
        ['', '', '', ''],
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training Technique':
        ['Offline', 'Offline', 'Offline', 'Offline'],   # Offline
    'A_ML task':
        ['Multiclass Classification', 'Multiclass Classification', 'Multiclass Classification', 'Multiclass Classification'],   # Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}

# inputs HPO BeliefRuleBase_v13 - TRANSPARENCY TESTING
inputs_HPO_BRB_TRANSPARENCY_v13 = {
    'A_UR: quality demands':
        ['', 'high', 'high', 'high'],
    'A_User\'s programming ability':
        ['', '', '', ''],
    'A_UR: need for model transparency':
        ['yes', 'yes', 'must', ''],
    'A_UR: Availability of a well documented library':
        ['', '', '', 'yes'],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: Number of workers/kernels for parallel computing':
        ['', '', '', ''],
    'A_Production application area':
        ['', '', '', ''],  # 'Predictive Quality'
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['7200.0:172800', '>172800', '7200.0:172800', '7200.0:172800'],  # >172800, '7200.0:172800'
    'A_Machine Learning Algorithm':
        ['XGBoost', 'XGBoost', 'XGBoost', 'XGBoost'],
    'A_Obtainability of good approximate':
        ['', '', '', ''],
    'A_Supports parallel evaluations':
        ['', '', '', ''],
    'A_Dimensionality of HPs':
        ['', '', '', ''],
    'A_Conditional HP space':
        ['', '', '', 'yes'],
    'A_HP datatypes':
        ['', '', '', ''],
    'A_Availability of a warm-start HP configuration':
        ['', '', '', ''],
    'A_Obtainability of gradients':
        ['', '', '', ''],
    'A_Input Data':
        ['', '', '', ''],  # Image data
    'A_#Instances training dataset':
        ['', '', '', ''],
    'A_Ratio training to test dataset':
        ['', '', '', ''],
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training Technique':
        ['Offline', 'Offline', 'Offline', 'Offline'],   # Offline
    'A_ML task':
        ['Multiclass Classification', 'Multiclass Classification', 'Multiclass Classification', 'Multiclass Classification'],   # Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}

# inputs HPO BeliefRuleBase_v13 - extreme input testing
""" 1) activation of an existing rule: is the outcome like the rule?
    2) very generic input that does not match any rule: does the BRBES still provide a rec?
    3) very specific input that does not match any rule: does the BRBES still provide a rec?
    4) uncertain input: {'low': 0.5, 'medium': 0.5} does it work?
    """
inputs_HPO_BRB_EXTREMEINPUT_v13 = {
    'A_UR: quality demands':
        ['high', '', '', 'high'],   # {'low': 1.0, 'medium': 1.0, 'high': 1.0}
    'A_User\'s programming ability':
        ['high', {'low': 0.5, 'medium': 0.5}, '', ''],
    'A_UR: need for model transparency':
        ['', '', '', ''],
    'A_UR: Availability of a well documented library':
        ['', '', '', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: Number of workers/kernels for parallel computing':
        ['', '', '', 'yes'],
    'A_Production application area':
        ['Predictive Quality', '', '', ''],  # 'Predictive Quality'
    'A_Number of maximum function evaluations/ trials budget':
        ['>100', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['1.0:86400', '<7200', '', '7200.0:172800'],  # >172800, '7200.0:172800'
    'A_Machine Learning Algorithm':
        ['', 'XGBoost', '', ''],
    'A_Obtainability of good approximate':
        ['no', '', '', ''],
    'A_Supports parallel evaluations':
        ['no', '', '', ''],
    'A_Dimensionality of HPs':
        ['', '', '', ''],
    'A_Conditional HP space':
        ['', '', '', 'yes'],
    'A_HP datatypes':
        ['[discrete, ordinal, nominal]', '', '', ''], # [discrete, ordinal, nominal]
    'A_Availability of a warm-start HP configuration':
        ['', '', '', ''],
    'A_Obtainability of gradients':
        ['no', '', '', ''],
    'A_Input Data':
        ['', '', '', ''],  # Image data
    'A_#Instances training dataset':
        ['', '', '', ''],
    'A_Ratio training to test dataset':
        ['', '', '', ''],
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training Technique':
        ['', '', '', ''],   # offline
    'A_ML task':
        ['', '', '', ''],   # Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}

'''
1.	Typical user input which perfectly matches an existing rule
2.	Uncertain user input: does the BRBES provide a recommendation and is it sound?
3.	Very generic input that does not perfectly match any rule in the knowledge base: does the BRBES still provide a sound recommendation?
4.	Very specific input that does not perfectly match any rule in the knowledge base: does the BRBES still provide a sound recommendation?

'''
# inputs ML BeliefRuleBase_v5
inputs_ML_BRB_v5 = {
    'A_UR: quality demands':
        ['', '', 'high', 'high'],
    'A_User\'s programming ability':
        ['low', 'low', 'high', ''],
    'A_UR: need for model transparency':
        ['yes', '', '', ''],
    'A_UR: robustness of the model':
        ['', '', '', ''],
    'A_UR: scalability of the model':
        ['', '', '', ''],
    'A_UR: Availability of a well documented library':
        ['yes', '', '', ''],
    'A_UR: HPO or use of default values?':
        ['', '', '', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: access to parallel computing?':
        ['no', '', 'yes', 'yes'],
    'A_Hardware: access to high performance computing?':
        ['no', '', 'yes', 'yes'],
    'A_Production application area':
        ['Predictive Quality', 'Predictive Quality', 'Predictive Quality', ''],  #Predictive Quality
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Number of kernels used':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['>172800', '<7200', '>172800', '7200.0:172800'],
    'A_Input Data':
        ['', '', '', ''],  # Image data, Tabular data
    'A_#Instances training dataset':
        ['', '', '', ''],       # >1000000
    'A_Ratio training to test dataset':
        ['', '', '', ''],       # 2.0:9
    'A_Feature characteristics':
        ['', '', '', ''],   # [continuous, discrete, nominal, timestamp]
    'A_Number of features':
        ['', '', '', ''],      # <100
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training Technique':
        ['', '', '', ''],   # offline
    'A_ML task':
        ['', '', '', ''],   # Multiclass Classification
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}

# inputs ML BeliefRuleBase_v6
inputs_ML_BRB_v6 = {
    'A_UR: quality demands':
        ['', '', 'high', 'high'],
    'A_User\'s programming ability':
        ['low', 'low', 'high', ''],
    'A_UR: need for model transparency':
        ['yes', '', '', ''],
    'A_UR: robustness of the model':
        ['', '', '', ''],
    'A_UR: scalability of the model':
        ['', '', '', ''],
    'A_UR: Availability of a well documented library':
        ['yes', '', '', ''],
    'A_UR: HPO or use of default values?':
        ['', '', '', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: access to high performance computing':
        ['no', '', 'yes', 'yes'],
    'A_Production application area':
        ['Predictive Quality', 'Predictive Quality', 'Predictive Quality', ''],  #Predictive Quality
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Number of kernels used':
        ['', '', '', ''],
    'A_Total Computing Time [s]':
        ['>172800', '<7200', '>172800', '7200.0:172800'],
    'A_Input Data':
        ['', '', '', ''],  # Image data, Tabular data
    'A_#Instances training dataset':
        ['', '', '', ''],       # >1000000
    'A_Ratio training to test dataset':
        ['', '', '', ''],       # 2.0:9
    'A_Feature datatypes':
        ['', '', '', ''],   # [continuous, discrete, nominal, timestamp]
    'A_Number of features':
        ['', '', '', ''],      # <100
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training technique':
        ['', '', '', ''],   # offline
    'A_ML task':
        ['Regression', 'Binary Classification', '', ''],   # 'Multiclass Classification', 'Binary Classification'
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}

# inputs ML BeliefRuleBase_v6 3 SCENARIOS
inputs_ML_BRB_3UCs_v6 = {
    'A_UR: quality demands':
        ['', '', 'high', ''],
    'A_User\'s programming ability':
        ['low', 'medium', 'high', ''],
    'A_UR: need for model transparency':
        ['must', '', '', ''],    # 'must'
    'A_UR: robustness of the model':
        ['', '', '', ''],
    'A_UR: scalability of the model':
        ['', '', '', ''],
    'A_UR: Availability of a well documented library':
        ['yes', '', '', ''],    # 'must'
    'A_UR: HPO or use of default values?':
        ['', 'Default values', '', ''],
    'A_UR: Computer operating system':
        ['', '', '', ''],
    'A_Hardware: access to high performance computing':
        ['no', 'yes', 'yes', ''],
    'A_Production application area':
        ['Predictive Quality', 'Predictive Quality', 'Predictive Quality', ''],  #Predictive Quality
    'A_Number of maximum function evaluations/ trials budget':
        ['', '', '', ''],
    'A_Running time per trial [s]':
        ['', '', '', ''],
    'A_Number of kernels used':
        ['1', '8', '8', ''],
    'A_Total Computing Time [s]':
        ['>172800', '<3600', '>172800', ''],
    'A_Input Data':
        ['Tabular data', 'Tabular data', 'Tabular data', ''],  # Image data, Tabular data
    'A_#Instances training dataset':
        ['35400', '35400', '35400', ''],       # >1000000
    'A_Ratio training to test dataset':
        ['4.0', '4.0', '4.0', ''],       # 2.0:9
    'A_Feature datatypes':
        ['[continuous, discrete, nominal]', '[continuous, discrete, nominal]', '[continuous, discrete, nominal]', ''],   # [continuous, discrete, nominal, timestamp]
    'A_Number of features':
        ['11', '11', '11', ''],      # <100
    'A_Noise in dataset':
        ['', '', '', ''],   # yes
    'A_Training technique':
        ['Offline', 'Offline', 'Offline', ''],   # Offline
    'A_ML task':
        ['Binary Classification', 'Binary Classification', 'Binary Classification', ''],   # 'Multiclass Classification', 'Binary Classification'
    'A_Detailed ML task':
        ['', '', '', ''],   # Image Recognition
}

curdir_path = '/Users/philippnoodt/VirtualBox_VMs/Win10/Win10_SharedFolder/MA/coding/Bruno/git/brb/'
filename = 'csv_ML_BeliefRuleBase_wKO_v9.csv_RefVals_AntImp-1Mglobscaled.csv'
            #'csv_HPO_BeliefRuleBase_wKO_v13.csv_RefVals_AntImp-1Mscaled.csv'
            #'csv_ML_BeliefRuleBase_wKO_v6.csv_RefVals_AntImp-1Mscaled.csv'
            #'csv_HPO_BeliefRuleBase_v12.csv_all_1.csv'
            #'csv_HPO_BeliefRuleBase_v12.csv_RefVals_AntImp-UVscaled.csv'
            #'csv_HPO_BeliefRuleBase_v12.csv_RefVals_AntImp-1Mscaled.csv'

if __name__ == "__main__":

    # create model from rules.csv
    model = csv2BRB('csv_rulebases/' + filename,
                    #'csv_rulebases/csv_ML_BeliefRuleBase_v5.csv_spec_refvals*ant_imp--scaled.csv',
                    #'csv_rulebases/csv_HPO_BeliefRuleBase_v11.csv_spec_refvals*ant_imp--scaled.csv',
                    antecedents_prefix='A_',
                    consequents_prefix='D_',
                    deltas_prefix='del_',
                    thetas='thetas')
    print('Model created')

    # test with random, existing inputs
    #random_existing_input(model, 10, incomplete=0.5, rec="ML algorithm")

    # test with custom inputs
    """
    rec determines recommendation type: 'HPO technique' or 'ML algorithm'
    show_top enables showing best top X of consequents: 'all' or integer value , show_top=10
    """
    custom_input(model, inputs_ML_BRB_3UCs_v6, rec='ML algorithm', show_top=11)    # or 'ML algorithm', 'HPO technique', 'all'

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