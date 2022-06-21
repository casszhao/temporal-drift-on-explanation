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

bigdf = pd.read_csv('all_tasks_all_selective.csv')
for i, name in enumerate(task_list):
    path = './' + str(name) + '/fresh_predictive_results.csv'
    print(' ---------', str(name))

    bert = bigdf[bigdf['Task']==str(name)][['BERT F1','FRESH F1','LSTM F1','KUMA F1','SPECTRA F1','Domain']]
    bert = bert.rename(columns={'BERT F1':'BERT','FRESH F1':'FRESH','LSTM F1':'LSTM','KUMA F1':'KUMA','SPECTRA F1':'SPECTRA'})
    #bert['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']   

    df = pd.read_csv(path)
    df = df[['mean-f1','Domain','attribute_name']]
    attribute_list = df['attribute_name']
    df["attribute_name"].replace({"scaled_attention": "scaled attention"}, inplace=True)
    grouper = df.groupby('attribute_name')
    df = pd.concat([pd.Series(v['mean-f1'].tolist(), name=k) for k, v in grouper], axis=1)
    df = df * 100
    df['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']
    df = pd.merge(bert, df, how='right', on=['Domain']) 
    print(df)

    if 'Amaz' in name:
        SUB_NAME = str(name)
    else:
        SUB_NAME = str(name).capitalize()


    if i < 2:
        axs[0, i].scatter(df['Domain'], df['LSTM'], label='LSTM', color='black')
        axs[0, i].scatter(df['Domain'], df['SPECTRA'], label='SPECTRA', color='dimgrey')
        axs[0, i].scatter(df['Domain'], df['KUMA'], label='HardKUMA', color='royalblue')
        axs[0, i].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', marker='x', color='crimson') 
        axs[0, i].scatter(df['Domain'], df['BERT'], label='BERT', color='brown', marker='x')
        axs[0, i].scatter(df['Domain'], df['deeplift'], label='DL', color='gold', marker='x')
        axs[0, i].scatter(df['Domain'], df['scaled attention'], color='darkorange', marker='x', label=r'$\alpha\nabla\alpha$')
        axs[0, i].set_xlabel(SUB_NAME,fontsize=xlabel_size)


    elif 4 > i > 1:
        axs[1, i-2].scatter(df['Domain'], df['LSTM'], label='LSTM', color='black')
        axs[1, i-2].scatter(df['Domain'], df['SPECTRA'], label='SPECTRA', color='dimgrey')
        axs[1, i-2].scatter(df['Domain'], df['KUMA'], label='HardKUMA', color='royalblue')
        axs[1, i-2].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', marker='x', color='crimson') 
        axs[1, i-2].scatter(df['Domain'], df['BERT'], label='BERT', color='brown', marker='x')
        axs[1, i-2].scatter(df['Domain'], df['deeplift'], label='DL', color='gold', marker='x')
        axs[1, i-2].scatter(df['Domain'], df['scaled attention'], color='darkorange', marker='x', label=r'$\alpha\nabla\alpha$')

        axs[1, i-2].set_xlabel(SUB_NAME,fontsize=xlabel_size)


    else:
        axs[2, i-4].scatter(df['Domain'], df['LSTM'], label='LSTM', color='black')
        axs[2, i-4].scatter(df['Domain'], df['SPECTRA'], label='SPECTRA', color='dimgrey')
        axs[2, i-4].scatter(df['Domain'], df['KUMA'], label='HardKUMA', color='royalblue')
        
        axs[2, i-4].scatter(df['Domain'], df['gradients'], label=r'FRESH($x\nabla x $)', marker='x', color='crimson') 
        axs[2, i-4].scatter(df['Domain'], df['deeplift'], label='FRESH(DL)', color='gold', marker='x')
        axs[2, i-4].scatter(df['Domain'], df['scaled attention'], color='darkorange', marker='x', label=r'FRESH($\alpha\nabla\alpha$)')
        axs[2, i-4].scatter(df['Domain'], df['BERT'], label='BERT', color='brown', marker='x')
        axs[2, i-4].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    
#fig.suptitle('Predictive Performance Comparison of Selective Rationalizations', fontsize=12)
plt.subplots_adjust(
    left=0.1,
    bottom=0.183, 
    right=0.924, 
    top=0.95, 
    wspace=0.388, 
    hspace=0.471,
    )
plt.legend(bbox_to_anchor=(-0.25, -0.4), loc='upper center', borderaxespad=0, fontsize=9.5,
            fancybox=True,ncol=3)
#plt.xticks(fontsize=xtick_size)
plt.show()
fig1 = plt.gcf()
fig.savefig('./fresh_compare_attributes.png', dpi=600)
