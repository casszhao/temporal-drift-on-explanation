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

## 直接使用了 dataset_stats.csv
#           key_results.csv
#           fulltext_similarity rationale_similarity
#           posthoc_results/****.json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
''' generate select-then-predict results together for all tasks'''
''' generate predictive results together for all tasks'''

use_acc = False

xlabel_size = 11
xtick_size = 9

task_list = ['agnews', 'xfact', 'factcheck', 'AmazDigiMu', 'AmazPantry', 'yelp'] #, 'AmazDigiMu', 'AmazInstr', 'AmazPantry'

plt.style.use('ggplot')
fig, axs = plt.subplots(3, 2, figsize=(6, 7), sharey=False, sharex=False)

marker_style = dict(color='tab:blue', linestyle=':', marker='d',
                    #markersize=15, markerfacecoloralt='tab:red',
                    )

bigtable_list = []
for i, name in enumerate(task_list):
    print('----------------------------')
    print(name)

    # Full_data = pd.read_json('./models/' + str(name) + '_full/bert_predictive_performances.json')
    # InDomain = pd.read_json('./models/' + str(name) + '/bert_predictive_performances.json')
    # path = os.path.join('./models/', str(name),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
    # OOD1 = pd.read_json(path)
    # path = os.path.join('./models/', str(name),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
    # OOD2 = pd.read_json(path)

    # if args.get_all_seeds_for_predictive:
    #     pass
    # else:
    #     Full_data = Full_data[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
    #     InDomain = InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
    #     OOD1 = OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
    #     OOD2 = OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]

    # OOD12 = (OOD1 + OOD2)/2
    # Full_data['Domain'] = 'Full'
    # InDomain['Domain'] = 'SynD'
    # OOD1['Domain'] = 'AsyD1'
    # OOD2['Domain'] = 'AsyD2'
    # #OOD12['Domain'] = 'AsyD1+2'
    

    # result = pd.concat([Full_data, InDomain, OOD1, OOD2], ignore_index=False, axis=1).T
    # #result.to_csv('saved_everything/' + str(args.dataset) + '/bert_predictive.csv')




    if use_acc == True:
        path = './' + str(name) + '/selective_results_acc.csv'
        df = pd.read_csv(path)
        print(df)
        df['Bert ACC'] = df['Bert ACC']*100
        df['FRESH ACC'] = df['FRESH ACC']*100
        df = df.rename(column={'Bert ACC':'BERT', 'FRESH ACC':'FRESH', 'KUMA ACC':'KUMA','LSTM ACC':'LSTM'})
    else:
        path = './' + str(name) + '/selective_results.csv'
        df = pd.read_csv(path)
        df['Bert F1'] = df['Bert F1']*100
        df['FRESH F1'] = df['FRESH F1']*100
        df = df.rename(column={'Bert F1':'BERT', 'FRESH F1':'FRESH', 'KUMA F1':'KUMA','LSTM F1':'LSTM'})

    if i == 3 or i == 4:
        SUB_NAME = str(name)
    else:
        SUB_NAME = str(name).capitalize()

    df['Task']=SUB_NAME
    bigtable_list.append(df)

    makersize = 60

    if i < 2:
        axs[0, i].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', s=makersize)
        axs[0, i].scatter(df['Domain'], df['FRESH'], label='FRESH(α∇α)', marker='x', s=makersize)
        axs[0, i].scatter(df['Domain'], df['SPECTRA'], label='SPECTRA')
        axs[0, i].scatter(df['Domain'], df['KUMA'], label='HardKUMA')
        axs[0, i].scatter(df['Domain'], df['LSTM'], label='LSTM')
        axs[0, i].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    elif i > 3:
        axs[2, i-4].scatter(df['Domain'], df['Bert F1'], label='BERT', marker='x', s=makersize)
        axs[2, i-4].scatter(df['Domain'], df['FRESH F1'], label='FRESH(α∇α)', marker='x', s=makersize)
        axs[2, i-4].scatter(df['Domain'], df['SPECTRA F1'], label='SPECTRA')
        axs[2, i-4].scatter(df['Domain'], df['KUMA F1'], label='HardKUMA')
        axs[2, i-4].scatter(df['Domain'], df['LSTM'], label='LSTM')
        axs[2, i-4].set_xlabel(SUB_NAME,fontsize=xlabel_size)
    else:
        axs[1, i-2].scatter(df['Domain'], df['Bert F1'], label='BERT', marker='x', s=makersize+4)
        axs[1, i-2].scatter(df['Domain'], df['FRESH F1'], label='FRESH(α∇α)', marker='x', s=makersize+4)
        axs[1, i-2].scatter(df['Domain'], df['SPECTRA F1'], label='SPECTRA')
        axs[1, i-2].scatter(df['Domain'], df['KUMA F1'], label='HardKUMA')
        axs[1, i-2].scatter(df['Domain'], df['LSTM F1'], label='LSTM')
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
fig.savefig('./selective_predictive.png', dpi=600)

# print(bigtable_list)
all_tasks = pd.concat(bigtable_list, ignore_index=False)
all_tasks.to_csv('all_tasks_all_selective.csv')
exit()

bigtable_list = []
for name in task_list:

    bert = pd.read_csv(str(name) + '/bert_predictive.csv')[['mean-f1', 'Domain']]
    posthoc = pd.read_csv(str(name) + '/posthoc_faithfulness.csv')

    merge = pd.merge(posthoc, bert, on = 'Domain')
    #merge = merge.rename(columns={merge.columns[1]: 'Task'},inplace=True)
    if 'Amaz' in str(name):
        merge['Task'] = str(name)
    else:
        merge['Task'] = str(name).capitalize()
    print(merge)
    merge.to_csv(str(name)+ '/posthoc_and_predictive.csv')
    bigtable_list.append(merge)

all_tasks = pd.concat(bigtable_list, ignore_index=False)
all_tasks.to_csv('all_tasks_all_posthoc.csv')







for i, task in enumerate(task_list):

    task_name = str(task)
    print('-----------' , task_name)
    task_index = i+1
    pwd = os.getcwd()
    path = os.path.join(pwd, str(task_name), 'key_results.csv')

    ### predictive
    predictive = pd.read_csv(path)
    predictive = predictive[['Domain', 'Bert F1', 'FRESH F1', 'KUMA F1', 'LSTM F1']]
    try:
        print('loop in try  4444  predictive  444')
        predictive.loc[predictive["Domain"] == "In Domain(Baseline)", "Domain"] = "InDomain"
        print(' In Domain(Baseline) changed to InDomain')
    except:
        print('no try')

    ### data_stat
    data_stat = pd.read_csv('./' + task_name + '/dataset_stats.csv')#.iloc[: , 1:]
    data_stat['InterTimeSpan(D)'] = pd.to_numeric(data_stat['InterTimeSpan(D)'].astype(str).str.replace(r' days$', '', regex=True))
    data_stat['TimeSpan(D)'] = pd.to_numeric(data_stat['TimeSpan(D)'].astype(str).str.replace(r' days$', '', regex=True))

    data_stat['TimeDensity'] = data_stat['Data Num']/data_stat['TimeSpan(D)']
    data_stat['InterTimeDensity'] = data_stat['Data Num']*0.5/data_stat['InterTimeSpan(D)']
    data_stat['Data Num'] = pd.to_numeric(data_stat['Data Num'])
    data_stat = data_stat[['Domain', 'Label distribution', 'Interquartile-Oldest',
       'Median', 'Interquartile-Newest', 'InterTimeSpan(D)', 'Oldest Date',
       'Newest Date', 'TimeSpan(D)', 'Data Num', 'TimeDensity','InterTimeDensity','TimeDiff']]

    for file in os.listdir('./' + task_name + '/'):
        if fnmatch.fnmatch(file, '*fulltext_similarity*'):
            full_similarity = pd.read_csv(task_name + '/' + file)
            full_similarity_JS_TOPIC = full_similarity[full_similarity['Rep_Mea'] == 'Topic jensen-shannon']
            full_similarity_JS_TOPIC = full_similarity_JS_TOPIC[['Similarity', 'Domain']]
            full_similarity_JS_TOPIC = full_similarity_JS_TOPIC.rename(columns={"Similarity": "Text_Simi"})
            try:
                print('loop in try')
                full_similarity_JS_TOPIC.loc[full_similarity_JS_TOPIC["Domain"] == "In Domain(Baseline)", "Domain"] = "InDomain"
                print(' In Domain(Baseline) changed to InDomain')
            except:
                print(' no try')
        elif fnmatch.fnmatch(file, '*rationale_similarity*'):
            rationales_similarity = pd.read_csv(task_name + '/' + file)
            rationales_SAtopk_similarity_JS_TOPIC = rationales_similarity[
                (rationales_similarity['Rep_Mea'] == 'Topic jensen-shannon') & (
                        rationales_similarity['threshold'] == 'topk') & (
                        rationales_similarity['attribute_name'] == 'scaled attention')]
            rationales_SAtopk_similarity_JS_TOPIC = rationales_SAtopk_similarity_JS_TOPIC[['Similarity', 'Domain']]
            rationales_SAtopk_similarity_JS_TOPIC = rationales_SAtopk_similarity_JS_TOPIC.rename(
                columns={"Similarity": "Rationales_Simi"})
            try:
                rationales_SAtopk_similarity_JS_TOPIC.loc[rationales_SAtopk_similarity_JS_TOPIC["Domain"] == "In Domain(Baseline)", "Domain"] = "InDomain"
                print(' In Domain(Baseline) changed to InDomain')
            except:
                print(' no try ')
        else:
            pass


    ##### posthoc
    def json2df(df, domain):
        df = df[['scaled attention']].rename(columns={'scaled attention': 'Posthoc'})

        list_of_list = []
        for col in range(0, len(df.columns)):  # the range length equals to the number of attributes, if remove ig
            rationales_sufficiency = df.iloc[0, col].get('mean')
            rationales_comprehensiveness = df.iloc[1, col].get('mean')
            rationales_AOPCsufficiency = df.iloc[2, col].get('mean')
            rationales_AOPCcomprehensiveness = df.iloc[3, col].get('mean')
            four_eval_metrics = [rationales_sufficiency, rationales_comprehensiveness,
                                 rationales_AOPCsufficiency, rationales_AOPCcomprehensiveness]
            list_of_list.append(four_eval_metrics)
        df_tf = pd.DataFrame.from_records(list_of_list).transpose()
        df_tf.columns = df.columns  # ['random','scaled_attention','attention','ig','lime','gradients','deeplift']

        df_tf['Rationales_metrics'] = ['Sufficiency', 'Comprehensiveness', 'AOPC_sufficiency', 'AOPC_comprehensiveness']
        df_tf['Domain'] = str(domain)
        df_tf = df_tf.set_index('Rationales_metrics')
        return df_tf


    for fname in os.listdir('../posthoc_results/' + str(task_name) + '/'):
        if 'topk' in fname and 'description.json' in fname:
            if 'ood1' in fname:
                ood1_path = os.path.join('../posthoc_results', str(task_name), fname)
            elif 'ood2' in fname:
                ood2_path = os.path.join('../posthoc_results', str(task_name), fname)
            else:
                indomain_path = os.path.join('../posthoc_results', str(task_name), fname)
    json = pd.read_json(indomain_path)
    df = json2df(json, 'InDomain')
    OOD1 = pd.read_json(ood1_path)
    df1 = json2df(OOD1, 'OOD1')
    OOD2 = pd.read_json(ood2_path)
    df2 = json2df(OOD2, 'OOD2')
    Posthoc = pd.concat([df, df1, df2], ignore_index=False)

    AOPC_suff = Posthoc.copy().loc['AOPC_sufficiency', :]
    AOPC_suff.rename(columns={"Posthoc": "AOPC_suff"}, inplace=True)
    AOPC_compr = Posthoc.copy().loc['AOPC_comprehensiveness', :]
    AOPC_compr.rename(columns={"Posthoc": "AOPC_compr"}, inplace=True)


    data_list = [predictive, AOPC_suff, AOPC_compr, full_similarity_JS_TOPIC, rationales_SAtopk_similarity_JS_TOPIC, data_stat]
    
    for df in data_list:
        print('-----------')
        print(df)

    bigtable = reduce(lambda left, right: pd.merge(left, right, on=['Domain'],
                                                   how='outer'), data_list)
    
    print(bigtable)
    bigtable.to_csv(task_name + '/bigtable.csv')
    bigtable_list.append(bigtable)

    bigtable_for_correlation = bigtable.dropna()
    print(bigtable_for_correlation)
    corr = bigtable_for_correlation.corr()
    cols_included = corr.index


    sns.set_theme(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(9, 9))

    # ax.set_xticks(np.arange(len(cols_included)), labels=cols_included, rotation=65, fontsize=11)
    # ax.set_yticks(np.arange(len(cols_included)), labels=cols_included, rotation=0, fontsize=11)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .7})
    plt.title(task_name, fontsize=17)
    plt.tight_layout()
    #plt.show()
    plt.savefig(str(task_name) + '.png', dpi=200, format='png')



bigtable_with = []
for i, task in enumerate(bigtable_list):
    df = bigtable_list[i]
    df['Task'] = str(task_list[i])
    bigtable_with.append(df)
bigtableof4 = pd.concat([bigtable_with[0], bigtable_with[1], bigtable_with[2], bigtable_with[3]], ignore_index=False)
bigtableof4.to_csv('./bigtable_of_alltasks.csv')
print('done')

#
#
# plt.subplots_adjust(wspace=0.1, hspace=1) #top=0.8, bottom=0.2, right=0.7, left=0
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)




