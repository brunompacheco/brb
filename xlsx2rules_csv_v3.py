import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------
# choose type of antecedent weights
# ['all 1', 'number of total ref_values', 'number of specific ref_values', 'antecedent importance',
# 'number of specific ref_values * antecedent importance']
delta_type = 'ref_values * antecedent importance'    # 'all 1', 'ref_values * antecedent importance'
scale_deltas = '1 Mean global'     # "1 mean", "unit variance", 'nope', '1 Mean global'
time_deltas = None        # 'use 1.0', 'use 0.33333', None

filename = 'HPO_BeliefRuleBase_wKO_v13.csv'  #excel_rulebases/
raw_filepath = os.path.join(os.curdir, 'excel_rulebases/' + filename)
excel_rulebase = pd.read_csv(raw_filepath, sep=';', header=None)

def get_number_of_rules(excel_rulebase):
    num_rules = 1
    while pd.notnull(excel_rulebase.iloc[num_rules+5, 0]):
        num_rules += 1
    return num_rules

def fill_in_antecedent_weights(rulebase, excel_rulebase, delta_type, scale_deltas,
                               num_rules, antecedents, antecedent_dict, time_deltas):

    _rulebase = rulebase
    _ant_weight_strat = ''
    if delta_type == 'all 1':
        _ant_weight_strat = _ant_weight_strat + 'all_1'
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                _rulebase.loc[rule, 'del_' + ant] = 1

    elif delta_type == 'number of antecedent occurrences':
        _ant_weight_strat = _ant_weight_strat + 'antecedent_occurrence'
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                if pd.notnull(excel_rulebase.iloc[rule+5, antecedent_dict[ant][0]]):
                    _rulebase.loc[rule, 'del_' + ant] = 1/antecedent_dict[ant][1].sum()

    elif delta_type == 'ref_values':
        _ant_weight_strat = _ant_weight_strat + 'RefVals'
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                ref_value = excel_rulebase.iloc[rule+4, antecedent_dict[ant][0]]
                if pd.notnull(ref_value):
                    x = antecedent_dict[ant][1].get(ref_value)
                    _rulebase.loc[rule, 'del_' + ant] = 1/antecedent_dict[ant][1].get(ref_value)

    elif delta_type == 'antecedent importance':
        _ant_weight_strat = _ant_weight_strat + 'AntImp'
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                _rulebase.loc[rule, 'del_' + ant] = 1 * float(antecedent_dict[ant][2])

    elif delta_type == 'ref_values * antecedent importance':
        _ant_weight_strat = _ant_weight_strat + 'RefVals_AntImp'
        for rule in range(1, num_rules + 1):
            for ant in antecedents[2:]:
                ref_value = excel_rulebase.iloc[rule+4, antecedent_dict[ant][0]]
                if pd.notnull(ref_value):
                    x = antecedent_dict[ant][1].get(ref_value)
                    a_i = float(antecedent_dict[ant][2])
                    denom = antecedent_dict[ant][1].get(ref_value)
                    _rulebase.loc[rule, 'del_' + ant] = float(antecedent_dict[ant][2]) \
                                                       / antecedent_dict[ant][1].get(ref_value)

    if scale_deltas == "unit variance":
        _ant_weight_strat = _ant_weight_strat + '-UVscaled'
        scaler = StandardScaler(with_mean=False)
        for rule in range(1, num_rules + 1):
            start, end = 'del_' + antecedents[2], 'del_' + antecedents[-1]
            deltas_rule = _rulebase.loc[rule, start:end].to_numpy().reshape(-1, 1)
            deltas_rule_scaled = scaler.fit_transform(deltas_rule).reshape(1, -1)
            _rulebase.loc[rule, start:end] = deltas_rule_scaled
    elif scale_deltas == "1 mean":
        _ant_weight_strat = _ant_weight_strat + '-1Mscaled'
        for rule in range(1, num_rules + 1):
            start, end = 'del_' + antecedents[2], 'del_' + antecedents[-1]
            deltas_rule = _rulebase.loc[rule, start:end].to_numpy().reshape(-1, 1)
            sum = np.nansum(np.array(deltas_rule.astype(float)))
            arr = ~np.isnan(np.array(deltas_rule.astype(float)))
            length = arr.sum() #.shape[0]
            deltas_rule_scaled = (deltas_rule*length/sum).reshape(1, -1)
            _rulebase.loc[rule, start:end] = deltas_rule_scaled
    elif scale_deltas == "1 Mean global": #normalize [0, 2]
        _ant_weight_strat = _ant_weight_strat + '-1Mglobscaled'
        minmaxscaler = MinMaxScaler(feature_range=(0, 2))
        start, end = 'del_' + antecedents[2], 'del_' + antecedents[-1]
        _df = _rulebase.loc[:, start:end]
        _df = _df * np.sum(_df.count()) / np.nansum(_df)
        _rulebase.loc[:, start:end] = _df
        #_rulebase.loc[:, start:end] = minmaxscaler.fit_transform(_rulebase.loc[:, start:end])
        #_rulebase.loc[:, start:end] = deltas_rule_scaled

    if time_deltas == 'use 0.33333':
        _ant_weight_strat = _ant_weight_strat + '-TIME1'
        cols = ['del_' + 'Number of maximum function evaluations/ trials budget',
                'del_' + 'Running time per trial [s]', 'del_' + 'Total Computing Time [s]']
        _rulebase[_rulebase.loc[:, cols].notnull()] = 0.33333
    elif time_deltas == 'use 1.0':
        _ant_weight_strat = _ant_weight_strat + '-TIME1'
        cols = ['del_' + 'Number of maximum function evaluations/ trials budget',
                'del_' + 'Running time per trial [s]', 'del_' + 'Total Computing Time [s]']
        _rulebase[_rulebase.loc[:, cols].notnull()] = 1.0

    return _rulebase, _ant_weight_strat


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

            # dictionary with the ant's column in the excel and a series of its ref_values,
            # their frequency and the antecedent importance
            antecedent_dict[excel_rulebase.iloc[3, counter]] = [counter]
            ref_values = excel_rulebase.iloc[5:5+num_rules, counter].value_counts()
            antecedent_dict[excel_rulebase.iloc[3, counter]].append(ref_values)
            antecedent_dict[excel_rulebase.iloc[3, counter]].append(excel_rulebase.iloc[3, counter+1])

        elif excel_rulebase.iloc[0, counter] == 'A_I':
            antecedent_importances.append(excel_rulebase.iloc[3, counter])

        elif excel_rulebase.iloc[0, counter] == 'D':
            consequents.append(excel_rulebase.iloc[3, counter])
            consequent_dict[excel_rulebase.iloc[3, counter]] = counter
        counter += 1

    # creating the rules dataframe which will be the final .csv
    thetas = ['thetas']
    A_list = ['A_' + ant for ant in antecedents[2:]]
    del_list = ['del_' + ant for ant in antecedents[2:]]
    D_list = ['D_' + con for con in consequents]
    A_I_list = ['Antecedent_Importance']
    csv_rulebase = pd.DataFrame(index=range(1, 1 + num_rules),
                                columns=(thetas + A_list + D_list + del_list + A_I_list))

    # filling the ruleid column
    for num in range(num_rules):
        csv_rulebase.iloc[num, 0] = num + 1

    # adding the rule weights: thetas
    for rule in range(1, num_rules+1):
        csv_rulebase.loc[rule, 'thetas'] = excel_rulebase.iloc[rule+4, 1]

    # adding the antecedent importances
    for idx, (ant, info) in enumerate(antecedent_dict.items()):
        csv_rulebase.loc[idx+1, 'Antecedent_Importance'] = ant + '_AI:_' + str(info[2])

    # filling in the rules
    for rule in range(1, num_rules+1):
        for ant in antecedents[2:]:
            csv_rulebase.loc[rule, 'A_' + ant] = excel_rulebase.iloc[rule+4, antecedent_dict[ant][0]]
            csv_rulebase.loc[rule, 'del_' + ant] = excel_rulebase.iloc[rule + 4, antecedent_dict[ant][0]+1]
        for con in consequents:
            belief = excel_rulebase.iloc[rule + 4, consequent_dict[con]]
            if isinstance(belief, str):
                belief = belief.replace(',','.')
            csv_rulebase.loc[rule, 'D_' + con] = belief

    csv_rulebase, ant_weight_strat = fill_in_antecedent_weights(csv_rulebase, excel_rulebase,
                                                  delta_type, scale_deltas, num_rules,
                                                  antecedents, antecedent_dict, time_deltas)

    rulebase_name = 'csv_' + filename + '_' + ant_weight_strat + '.csv'
    csv_rulebase.to_csv(rulebase_name)
    print('done.')
    print('really done')


