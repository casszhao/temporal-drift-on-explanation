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

bigtable_list = []
task_list = ['complain', 'binarybragging', 'xfact', 'factcheck']
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
    plt.savefig(str(task_name) + '.png', dpi=600, format='png')



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




