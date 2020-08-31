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

def custom_input(A_i, X_i):
    user_inputs = A_i.copy()
    for idx, ref_val in enumerate(A_i):
        if ref_val == X_i:
            user_inputs[idx] = 1
        else:
            user_inputs[idx] = 0
    return user_inputs

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


    # create individual test inputs
    print('\nindividual test inputs')
    # individual test input klein2019_rules.csv
    attr_input = dict()
    user_inputs = []
    for U_i, A_i in model.A.items():
        if U_i == "Dimensionality of HPs":
            user_inputs = custom_input(A_i, 2)
        if U_i == "Conditional HP Space":
            user_inputs = custom_input(A_i, 'no')
        if U_i == "#continuous HPs of ML alg.":
            user_inputs = custom_input(A_i, '0')
        if U_i == "Number of possible function evaluations/maximum number of trials":
            user_inputs = custom_input(A_i, '<50')
        if U_i == "Machine Learning Algorithm":
            user_inputs = custom_input(A_i, 'SVM')
        if U_i == "Dataset to perform ML task":
            user_inputs = custom_input(A_i, '16 OpenML classification datasets')
        if U_i == "Artificial noise in dataset":
            user_inputs = custom_input(A_i, 'no')
        if U_i == "Surrogate Benchmarking":
            user_inputs = custom_input(A_i, 'SVM')
        if U_i == "Task that was performed by the ML algorithm who's HPs were optimized":
            user_inputs = custom_input(A_i, 'Classification')

        attr_input[U_i] = dict(zip(A_i, user_inputs))

    X = AttributeInput(attr_input)
    belief_degrees = model.run(X)
    print(attr_input)
    print(tuple(zip(model.D, belief_degrees)))
    '''
    # single values
    attr_input = dict()
    for U_i, A_i in model.A.items():
        if any('<' in ref_val for ref_val in A_i) == True:
            rand_ref_val = random.choice(A_i)
            if '<' in rand_ref_val:
                rand_ref_val = ''.join(filter(str.isdigit, rand_ref_val))
                rand_ref_val = random.randint(0, int(rand_ref_val))
                
                and now I have the problem, that I cannot use the user_inputs array since the input doesn't
                match an entry in A_i for 100% and so I could enter a 1
                
        user_inputs = A_i.copy()
        for idx, ref_val in enumerate(A_i):
            if ref_val == rand_ref_val:
                user_inputs[idx] = 1
            else:
                user_inputs[idx] = 0
        attr_input[U_i] = dict(zip(A_i, user_inputs))

    X = AttributeInput(attr_input)
    belief_degrees = model.run(X)
    print(belief_degrees)
    '''


    print('success')