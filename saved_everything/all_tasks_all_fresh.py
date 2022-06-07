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
fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharey=False, sharex=False)

marker_style = dict(color='tab:blue', linestyle=':', marker='d',
                    #markersize=15, markerfacecoloralt='tab:red',
                    )

for i, name in enumerate(task_list):
    path = './' + str(name) + '/fresh_predictive_results.csv'
    df = pd.read_csv(path)
    # df['Task']=str(name).capitalize()
    print(df)

    if i == 3 or i == 4:
        SUB_NAME = str(name)
    else:
        SUB_NAME = str(name).capitalize()

    makersize = 60

    if i < 2:
        axs[0, i].scatter(df['Domain'], df[''], label='Gradients') #, marker='x', s=makersize
        axs[0, i].scatter(df['Domain'], df['deeplift'], label='Deeplift')
        axs[0, i].scatter(df['Domain'], df['scaled attention'], label='Scaled attention')
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
        axs[1, i-2].scatter(df['Domain'], df['deeplift'], label='Deeplift')
        axs[1, i-2].scatter(df['Domain'], df['scaled attention'], label='Scaled attention')
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
plt.xticks(fontsize=xtick_size)
plt.show()
fig1 = plt.gcf()
fig.savefig('./selective_predictive.png', dpi=250)