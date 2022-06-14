from turtle import color
import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mticker
import fnmatch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


task_list = ['agnews', 'xfact', 'factcheck', 'AmazPantry', 'AmazPantry', 'yelp'] # 'AmazDigiMu', 'AmazPantry', 

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
fig, axs = plt.subplots(3, 2, figsize=(4.3, 7), sharey=False, sharex=False)

marker_style = dict(color='tab:blue', linestyle=':', marker='d',
                    #markersize=15, markerfacecoloralt='tab:red',
                    )
xlabel_size = 12
xtick_size = 22
ytick_size = 22
makersize = 66
for i, name in enumerate(task_list):
    path = './' + str(name) + '/fresh_predictive_results.csv'
    bert_path = './' + str(name) + '/bert_predictive.csv'
    bert = pd.read_csv(bert_path)[['mean-f1', 'Domain']]
    bert = bert.rename(columns={'mean-f1':'BERT'})
    bert['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']   

    df = pd.read_csv(path)
    df = df[['mean-f1','Domain','attribute_name']]
    attribute_list = df['attribute_name']
    df["attribute_name"].replace({"scaled_attention": "scaled attention"}, inplace=True)
    #print(df)
    grouper = df.groupby('attribute_name')
    df = pd.concat([pd.Series(v['mean-f1'].tolist(), name=k) for k, v in grouper], axis=1)
    df['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']
    df['BERT'] = bert['BERT']  
    print(df)

    if 'Amaz' in name:
        SUB_NAME = str(name)
    else:
        SUB_NAME = str(name).capitalize()


    if i < 2:
        axs[0, i].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', color='dimgrey') #, marker='x', s=makersize
        axs[0, i].scatter(df['Domain'], df['deeplift'], label='DL', marker='d', color='darkorange')
        axs[0, i].scatter(df['Domain'], df['scaled attention'], label=r'$\alpha\nabla\alpha$', marker='<', color='steelblue')
        axs[0, i].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', color='red')
        axs[0, i].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    elif i > 3:
        axs[2, i-4].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', color='dimgrey')
        axs[2, i-4].scatter(df['Domain'], df['deeplift'], label='DL', marker='d', color='darkorange')
        axs[2, i-4].scatter(df['Domain'], df['scaled attention'], label=r'$\alpha\nabla\alpha$', marker='<', color='steelblue')
        axs[2, i-4].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', color='red')
        axs[2, i-4].set_xlabel(SUB_NAME,fontsize=xlabel_size)

    else:
        axs[1, i-2].scatter(df['Domain'], df['gradients'], label='$x\nabla x $', color='dimgrey')
        axs[1, i-2].scatter(df['Domain'], df['deeplift'], label='DL', marker='d', color='darkorange')
        axs[1, i-2].scatter(df['Domain'], df['scaled attention'], label='Scaled attention', marker='<', color='steelblue')
        axs[1, i-2].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', color='red')
        axs[1, i-2].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    
#fig.suptitle('Predictive Performance Comparison of Selective Rationalizations', fontsize=12)
plt.subplots_adjust(
    left=0.1,
    bottom=0.126, 
    right=0.963, 
    top=0.995, 
    wspace=0.388, 
    hspace=0.471,
    )
plt.legend(bbox_to_anchor=(-0.19, -0.4), loc='upper center', borderaxespad=0, fontsize=10,
            fancybox=True,ncol=5)
#plt.xticks(fontsize=xtick_size)
plt.show()
fig1 = plt.gcf()
fig.savefig('./fresh_compare_attributes.png', dpi=600)