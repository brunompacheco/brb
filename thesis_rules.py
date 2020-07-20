import os

import numpy as np
import pandas as pd
from brb.brb import RuleBaseModel, AttributeInput


if __name__ == "__main__":
    rules_filepath = os.path.join(os.curdir, 'hanno_rules.csv')

    rules = pd.read_csv(rules_filepath, sep=';', index_col=0)

    rules = rules[rules.columns[1:]]  # drop weight column (empty)

    consequent = rules.iloc[:,-1]
    antecedents = rules.iloc[:,:-1]

    # model setup
    U = antecedents.columns
    D = consequent.unique()
    A = {U_i: antecedents[U_i].dropna().unique() for U_i in U}
    model = RuleBaseModel(U=U, A=A, D=D)

    # add rules
    A_ks = np.matrix(antecedents.values)
    model.add_rules_from_matrix(
        A_ks=A_ks,
        #A_ks=antecedents.values,
        betas=consequent.values
    )
    
    # TODO: add tests
