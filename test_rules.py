import os
import random
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

    # create test inputs

    attr_input = dict()
    for U_i, A_i in model.A.items():
        print('Input for {} {}:'.format(U_i, A_i))

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

    print(belief_degrees)


    print('success')