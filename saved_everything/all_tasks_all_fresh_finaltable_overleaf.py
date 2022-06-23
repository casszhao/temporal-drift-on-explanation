from turtle import color
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.ticker as mticker
import fnmatch
import os
import numpy as np


task_list = ['agnews', 'xfact', 'factcheck', 'AmazDigiMu', 'AmazPantry', 'yelp'] # 'AmazDigiMu', 'AmazPantry', 

# for i, task in enumerate(task_list):
#         if i == 0:
#             df = pd.read_csv(str(task) +'/fresh_predictive_results.csv')
#             df['Task'] = str(task)
#         else:
#             temp_df = pd.read_csv(str(task) +'/fresh_predictive_results.csv')
#             temp_df['Task'] = str(task)
#             df = pd.concat([df, temp_df])
# print(df)       

# df.to_csv('all_tasks_all_fresh_results.csv')

plt.style.use('ggplot')
fig, axs = plt.subplots(3, 2, figsize=(4.6, 7), sharey=False, sharex=False)

marker_style = dict(color='tab:blue', linestyle=':', marker='d',
                    #markersize=15, markerfacecoloralt='tab:red',
                    )
xlabel_size = 12
xtick_size = 22
ytick_size = 22
makersize = 66

three_fresh_for_overleaf_appendix = []
bigdf = pd.read_csv('all_tasks_all_selective.csv')
for i, name in enumerate(task_list):
    path = './' + str(name) + '/fresh_predictive_results.csv'
    print(' ---------', str(name))

    bert = bigdf[bigdf['Task']==str(name)][['BERT F1','BERT std', 'FRESH F1','LSTM F1','KUMA F1','SPECTRA F1','Domain']]
    # bert = bert.rename(columns={'BERT F1':'BERT','FRESH F1':'FRESH','LSTM F1':'LSTM','KUMA F1':'KUMA','SPECTRA F1':'SPECTRA'})

    df = pd.read_csv(path)
    df = df[['mean-f1','std-f1', 'Domain','attribute_name']]
    df["attribute_name"].replace({"scaled_attention": "scaled attention"}, inplace=True)
    grouper = df.groupby('attribute_name')

    df1 = pd.concat([pd.Series(v['mean-f1'].tolist(), name=k) for k, v in grouper], axis=1)
    df1 = df1 * 100
    print(df1)
    df1 = df1.rename(columns = {'scaled attention': 'scaled attention f1', 'gradients': 'gradients f1', 'deeplift': 'deeplift f1'})
    df1['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']
    print(df1)

    df2 = pd.concat([pd.Series(v['std-f1'].tolist(), name=k) for k, v in grouper], axis=1)
    df2 = df2 * 100
    df2['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']
    print(df2)


    df3 = pd.merge(df2, df1, how='right', on=['Domain']) 
    print(df3)
    
    final = pd.merge(bert, df3, how='right', on=['Domain'])
    final['Task'] = str(name)
    final = final[['Task', 'Domain', 'BERT F1', 'BERT std', 
                                                            'scaled attention f1', 'scaled attention', 
                                                            'deeplift f1', 'deeplift', 
                                                            'gradients f1', 'gradients', 
                                                                ]]

    three_fresh_for_overleaf_appendix.append(final)
three_fresh_final = pd.concat(three_fresh_for_overleaf_appendix)
three_fresh_final.to_csv('three_fresh_final.csv')

