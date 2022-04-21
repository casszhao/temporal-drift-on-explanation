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


task_list = ['complain', 'binarybragging', 'xfact', 'factcheck']

for i, task in enumerate(task_list):

    task_name = str(task)
    print('-----------' , task_name)
    task_index = i+1

    predictive = pd.read_csv('./' + task_name + '/key_results.csv')
    predictive = predictive[['Domain', 'Bert F1', 'FRESH F1', 'KUMA F1', 'LSTM F1']]

    data_stat = pd.read_csv('./' + task_name + '/dataset_stats.csv').iloc[: , 1:]
    if task_name == 'binarybragging':
        data_stat['InterTimeSpan(D)'] = data_stat['Interquartile Time Span in Days']
        data_stat['TimeSpan(D)'] = data_stat['Time Span in Days']
        data_stat = data_stat.drop(['Interquartile Time Span in Days', 'Time Span in Days'], axis=1)

    else:
        data_stat['InterTimeSpan(D)'] = pd.to_numeric(data_stat['Interquartile Time Span in Days'].astype(str).str.replace(r' days$', '', regex=True))
        data_stat['TimeSpan(D)'] = pd.to_numeric(data_stat['Time Span in Days'].astype(str).str.replace(r' days$', '', regex=True))

    data_stat['TimeDensity'] = data_stat['Data Num']/data_stat['TimeSpan(D)']
    data_stat['InterTimeDensity'] = data_stat['Data Num']*0.5/data_stat['InterTimeSpan(D)']
    # data_stat['Data Num'] = pd.to_numeric(data_stat['Data Num'])

    for file in os.listdir('./' + task_name + '/'):
        if fnmatch.fnmatch(file, '*fulltext_similarity*'):
            full_similarity = pd.read_csv(task_name + '/' + file)
            full_similarity_JS_TOPIC = full_similarity[full_similarity['Rep_Mea'] == 'Topic jensen-shannon']
            full_similarity_JS_TOPIC = full_similarity_JS_TOPIC[['Similarity', 'Domain']]
            full_similarity_JS_TOPIC = full_similarity_JS_TOPIC.rename(columns={"Similarity": "Text_Simi"})
        elif fnmatch.fnmatch(file, '*rationale_similarity*'):
            rationales_similarity = pd.read_csv(task_name + '/' + file)
            rationales_SAtopk_similarity_JS_TOPIC = rationales_similarity[
                (rationales_similarity['Rep_Mea'] == 'Topic jensen-shannon') & (
                        rationales_similarity['threshold'] == 'topk') & (
                        rationales_similarity['attribute_name'] == 'scaled attention')]
            rationales_SAtopk_similarity_JS_TOPIC = rationales_SAtopk_similarity_JS_TOPIC[['Similarity', 'Domain']]
            rationales_SAtopk_similarity_JS_TOPIC = rationales_SAtopk_similarity_JS_TOPIC.rename(
                columns={"Similarity": "Rationales_Simi"})
        else:
            pass

    data_list = [predictive, full_similarity_JS_TOPIC, rationales_SAtopk_similarity_JS_TOPIC, data_stat]
    bigtable = reduce(lambda left, right: pd.merge(left, right, on=['Domain'],
                                                   how='outer'), data_list)
    print(bigtable)
    bigtable.to_csv(task_name + '/bigtable.csv')

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
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7})
    plt.title(task_name, fontsize=17)
    plt.tight_layout()
    plt.show()




    # fig, ax = plt.subplots(figsize=(8, 8))
    # im = ax.imshow(corr, interpolation='nearest')
    # fig.colorbar(im, orientation='vertical', fraction=0.05)
    #
    # # ax1 = plt.subplot(figsize=(8,8))
    # ax.set_xticks(np.arange(len(cols_included)), labels=cols_included, rotation=65, fontsize=11)
    # ax.set_yticks(np.arange(len(cols_included)), labels=cols_included, rotation=0, fontsize=11)
    # ax.title.set_text(task_name, fontsize=32)


#     handles, labels = ax1.get_legend_handles_labels()
#     ax1.legend(handles, labels, loc='upper center')
#     # plt.colorbar()
#     plt.imshow(corr)
#
#
# plt.subplots_adjust(wspace=0.1, hspace=1) #top=0.8, bottom=0.2, right=0.7, left=0
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.colorbar(cax=cax)




