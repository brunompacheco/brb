import numpy as np
import pandas as pd
import os

filename = 'BeliefRuleBase.csv'
raw_filepath = os.path.join(os.curdir, filename)

raw_rulebase = pd.read_csv(raw_filepath, sep=';')
counter = 1
antecedents = ['rule_id','rule_weight']
test = raw_rulebase.isnull()

# appending the antecedent names to the antecedents list
while raw_rulebase.iloc[2, counter] != 'Rule ID':
    if pd.notnull(raw_rulebase.iloc[2, counter]):
            antecedents.append('A_' + raw_rulebase.iloc[2, counter])
    counter += 1

# column id where the consequents start
idx_ruleid = counter

# creating a list with all consequent names
counter += 1
consequents = []
while counter < raw_rulebase.shape[1]:
    if pd.isnull(raw_rulebase.iloc[2, counter]):
        break
    consequents.append('D_' + raw_rulebase.iloc[2, counter])
    counter += 1

#extract belief table from the xlsx file
beliefs = raw_rulebase.iloc[3:, (idx_ruleid + 1):(idx_ruleid + 1 + len(consequents))].dropna(how='all')

num_rules = beliefs.shape[0]

#replace , with . in belief table for python float readability
for col in range(beliefs.shape[1]):
    for idx in range(beliefs.shape[0]):
        value = beliefs.iloc[idx, col]
        if pd.isna(value):
            continue
        elif ',' in value:
            beliefs.iloc[idx, col] = value.replace(',', '.')

# creating the rules dataframe which will be the final .csv
rules = pd.DataFrame(index=range(num_rules), columns=(antecedents + consequents))

# filling the ruleid column
for num in range(num_rules):
    rules.iloc[num, 0] = num + 1

# iterating through the antecedents to enter referential value in correct row (rule) in rules dataframe
for ant in antecedents[2:]:
    col = 0
    while col < idx_ruleid:
        anti = raw_rulebase.iloc[2, col]
        if raw_rulebase.iloc[2, col] == ant.split('_')[1]:
            x = 1
            while pd.notnull(raw_rulebase.iloc[2+x, col]):
                if pd.isnull(raw_rulebase.iloc[2+x, col+1]):
                    x += 1
                elif isinstance(raw_rulebase.iloc[2+x, col+1], float):
                    ruleid = raw_rulebase.iloc[2+x, col+1]
                    rules[ant].iloc[int(ruleid) - 1] = raw_rulebase.iloc[2 + x, col]
                    break
                else:
                    rule_list = raw_rulebase.iloc[2+x, col+1].split(',')
                    rule_list = [int(elem) for elem in rule_list]

                    for ruleid in rule_list:
                        rules[ant].iloc[ruleid - 1] = raw_rulebase.iloc[2 + x, col]
                    '''
                    data = []
                    for idx in range(beliefs.shape[0]):
                        if idx in rule_list:
                            data.append(raw_rulebase.iloc[2 + x, col])
                        else:
                            data.append('')
                        #ii = rules[ant]
                        #iii = ruleid
                        #rules[ant].iloc[int(ruleid)-1] = raw_rulebase.iloc[2 + x, col]
                    rules[ant] = data
                    '''
                    x += 1
        col += 1

for idx in range(beliefs.shape[0]):
    for col in range(beliefs.shape[1]):
        rules.iloc[idx, len(antecedents)+col] = beliefs.iloc[idx, col]

rules.to_csv('rulebase.csv')
print('done.')
print('really done')