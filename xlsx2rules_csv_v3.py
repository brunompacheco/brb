import numpy as np
import pandas as pd
import os
import math

# ---------------------------------------------------------------------
# choose type of antecedent weights
# ['all 1', 'num of total ref_values', 'num of specific ref_values']
delta_type = 'num of specific ref_values'

filename = 'excel_rulebases/20201001_HPO_BeliefRuleBase_v7.csv'
raw_filepath = os.path.join(os.curdir, filename)
excel_rulebase = pd.read_csv(raw_filepath, sep=';', header=None)

def get_number_of_rules(excel_rulebase):
    num_rules = 1
    while pd.notnull(excel_rulebase.iloc[num_rules+5, 0]):
        num_rules += 1
    return num_rules

def fill_in_antecedent_weights(rulebase, excel_rulebase, type, num_rules, antecedents, antecedent_dict):
    if type == 'all 1':
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                rulebase.loc[rule, 'del_' + ant] = 1

    elif type == 'num of total ref_values':
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                if pd.notnull(excel_rulebase.iloc[rule+5, antecedent_dict[ant][0]]):
                    rulebase.loc[rule, 'del_' + ant] = 1/antecedent_dict[ant][1].sum()

    elif type == 'num of specific ref_values':
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                idx = num_rules+4
                col = antecedent_dict[ant][0]
                ref_value = excel_rulebase.iloc[rule+4, antecedent_dict[ant][0]]
                if pd.notnull(ref_value):
                    x = antecedent_dict[ant][1].get(ref_value)
                    print('xx')
                    rulebase.loc[rule, 'del_' + ant] = 1/antecedent_dict[ant][1].get(ref_value)


if __name__ == "__main__":

    counter = 1
    antecedents = ['rule_id', 'rule_weight']
    antecedent_dict = {}
    referential_values = {}
    antecedent_importances = []
    consequents = []
    consequent_dict = {}
    deltas = []
    test = excel_rulebase.isnull()

    # get number of rules in excel
    num_rules = get_number_of_rules(excel_rulebase)

    # create lists with antecedent names, their importances and consequent names
    counter = 1
    while excel_rulebase.iloc[0, counter] != 'X':
        if excel_rulebase.iloc[0, counter] == 'A':
            antecedents.append(excel_rulebase.iloc[3, counter])

            # dictionary with the ant's column in the excel and a series of its ref_values
            # and their frequency
            antecedent_dict[excel_rulebase.iloc[3, counter]] = [counter]
            ref_values = excel_rulebase.iloc[5:5+num_rules, counter].value_counts()
            antecedent_dict[excel_rulebase.iloc[3, counter]].append(ref_values)

        elif excel_rulebase.iloc[0, counter] == 'A_I':
            antecedent_importances.append(excel_rulebase.iloc[3, counter])

        elif excel_rulebase.iloc[0, counter] == 'D':
            consequents.append(excel_rulebase.iloc[3, counter])
            consequent_dict[excel_rulebase.iloc[3, counter]] = counter
        counter += 1

    # creating the rules dataframe which will be the final .csv
    A_list = ['A_' + ant for ant in antecedents]
    del_list = ['del_' + ant for ant in antecedents]
    D_list = ['D_' + con for con in consequents]
    A_I_list = ['Antecedent_Importance']
    csv_rulebase = pd.DataFrame(index=range(1, 1 + num_rules),
                                columns=(A_list + D_list + del_list + A_I_list))

    # filling the ruleid column
    for num in range(num_rules):
        csv_rulebase.iloc[num, 0] = num + 1

    # filling in the rules
    for rule in range(1, num_rules+1):
        for ant in antecedents[2:]:
            csv_rulebase.loc[rule, 'A_' + ant] = excel_rulebase.iloc[rule+4, antecedent_dict[ant][0]]
            csv_rulebase.loc[rule, 'del_' + ant] = excel_rulebase.iloc[rule + 4, antecedent_dict[ant][0]+1]
        for con in consequents:
            csv_rulebase.loc[rule, 'D_' + con] = excel_rulebase.iloc[rule + 4, consequent_dict[con]]

    fill_in_antecedent_weights(csv_rulebase, excel_rulebase, delta_type, num_rules, antecedents, antecedent_dict)


    csv_rulebase.to_csv('hpo_rulebase_v7.csv')
    print('done.')
    print('really done')


