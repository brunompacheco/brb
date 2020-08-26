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

curdir_path = '/Users/philippnoodt/VirtualBox_VMs/Win10/Win10_SharedFolder/MA/coding/Bruno/git/brb/'

if __name__ == "__main__":
    rules_filepath = os.path.join(os.curdir, 'eggensperger2015_rules.csv')
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
        print(tuple(zip(model.D, belief_degrees)))


    # create random test inputs using new referential values
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