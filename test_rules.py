import os
import random
import string
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
    rules_filepath = os.path.join(os.curdir, 'klein2019_rules.csv')
    model = get_model_from_csv(rules_filepath)

    # create random test inputs using referential values identical with those in the rule base

    num_tests = 100
    counter = 0
    while counter < num_tests:
        counter += 1

        attr_input = dict()
        for U_i, A_i in model.A.items():
            rand_ref_val = random.choice(A_i)
            user_inputs = A_i.copy()
            for idx, ref_val in enumerate(A_i):
                if ref_val == rand_ref_val:
                    user_inputs[idx] = 1
                else:
                    user_inputs[idx] = 0
            attr_input[U_i] = dict(zip(A_i, user_inputs))

        X = AttributeInput(attr_input)
        belief_degrees = model.run(X)
        print(attr_input)
        print(tuple(zip(model.D, belief_degrees)))


    # create random test inputs using new referential values
    print('\nindividual test inputs')
    custom_input(inputs)


    print('success')