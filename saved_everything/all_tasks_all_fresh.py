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


task_list = ['agnews', 'xfact', 'factcheck', 'yelp'] # 'AmazDigiMu', 'AmazPantry', 

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
fig, axs = plt.subplots(2, 2, figsize=(6, 7), sharey=False, sharex=False)

marker_style = dict(color='tab:blue', linestyle=':', marker='d',
                    #markersize=15, markerfacecoloralt='tab:red',
                    )
xlabel_size = 22
xtick_size = 22
makersize = 60
for i, name in enumerate(task_list):
    path = './' + str(name) + '/fresh_predictive_results.csv'
    df = pd.read_csv(path)
    df = df[['mean-f1','Domain','attribute_name']]
    attribute_list = df['attribute_name']
    df["attribute_name"].replace({"scaled_attention": "scaled attention"}, inplace=True)
    print(df)
    grouper = df.groupby('attribute_name')
    df = pd.concat([pd.Series(v['mean-f1'].tolist(), name=k) for k, v in grouper], axis=1)
    df['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']   
    print(df)

    if i == 3 or i == 4:
        SUB_NAME = str(name)
    else:
        SUB_NAME = str(name).capitalize()


    if i < 2:
        axs[0, i].scatter(df['Domain'], df['gradients'], label='Gradients') #, marker='x', s=makersize
        axs[0, i].scatter(df['Domain'], df['deeplift'], label='Deeplift', marker='d')
        axs[0, i].scatter(df['Domain'], df['scaled attention'], label='Scaled attention', marker='x')
        axs[0, i].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    # elif i > 3:
    #     axs[2, i-4].scatter(df['Domain'], df['Bert F1'], label='BERT', marker='x', s=makersize)
    #     axs[2, i-4].scatter(df['Domain'], df['FRESH F1'], label='FRESH(α∇α)', marker='x', s=makersize)
    #     axs[2, i-4].scatter(df['Domain'], df['SPECTRA F1'], label='SPECTRA')
    #     axs[2, i-4].scatter(df['Domain'], df['KUMA F1'], label='HardKUMA')
    #     axs[2, i-4].scatter(df['Domain'], df['LSTM F1'], label='LSTM')
    #     axs[2, i-4].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    else:
        axs[1, i-2].scatter(df['Domain'], df['gradients'], label='Gradients')
        axs[1, i-2].scatter(df['Domain'], df['deeplift'], label='Deeplift', marker='d')
        axs[1, i-2].scatter(df['Domain'], df['scaled attention'], label='Scaled attention', marker='x')
        axs[1, i-2].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    
#fig.suptitle('Predictive Performance Comparison of Selective Rationalizations', fontsize=12)
plt.subplots_adjust(
    left=0.057,
    bottom=0.093, 
    right=0.78, 
    top=0.983, 
    wspace=0.274, 
    hspace=0.388,
    )
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=9)
#plt.xticks(fontsize=xtick_size)
plt.show()
fig1 = plt.gcf()
fig.savefig('./fresh_compare_attributes.png', dpi=250)