# 1. bert predictive resultes -- on In domain / ood1 / ood2
# 2. faithful: for both top / contigious -- on In domain / ood1 / ood2
# 3. FRESH results
# 4. kuma results (another script)
# 5. domain similarity between:  In domain / ood1 / ood2
# 6. rationale similarity between:  In domain / ood1 / ood2

# 7. datasets metadata: train/test/ size, time span, label distribution

from dataclasses import replace
from telnetlib import PRAGMA_HEARTBEAT
import pandas as pd
import json
import csv
import config.cfg
import os
import numpy as np
import argparse
import fnmatch
import seaborn as sns
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "factcheck",
)
parser.add_argument(
    '--get_all_seeds_for_predictive',
    help='get all seeds results for bert prediction',
    action='store_true',
    default=False
)
parser.add_argument(
    '--save_for_bert',
    help='decide which parts are in need',
    action='store_true',
    default=False
)
parser.add_argument(
    '--save_for_fresh',
    help='decide which parts are in need',
    action='store_true',
    default=False
)
parser.add_argument(
    '--save_for_kuma_lstm',
    help='decide which parts are in need',
    action='store_true',
    default=False
)
parser.add_argument(
    '--save_posthoc',
    help='decide which parts are in need',
    action='store_true',
    default=False
)
parser.add_argument(
    '--save_data_stat',
    help='decide which parts are in need',
    action='store_true',
    default=False
)
parser.add_argument(
    '--plot_time_distribution',
    help='decide which parts are in need',
    action='store_true',
    default=False
)

parser.add_argument(
    '--combine_all_data_stat',
    help='decide which parts are in need',
    action='store_true',
    default=False
)

parser.add_argument(
    '--save_posthoc_for_analysis',
    help='decide which parts are in need',
    action='store_true',
    default=False
)

parser.add_argument(
    '--plot_radar',
    help='decide which parts are in need',
    action='store_true',
    default=False
)


parser.add_argument(
    '--predictive_and_posthoc',
    help='decide which parts are in need',
    action='store_true',
    default=False
)


parser.add_argument(
    '--save_all_selective_rationales',
    help='decide which parts are in need',
    action='store_true',
    default=False
)

args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)


    
######################## plot time distribution

from datetime import datetime


def df2stat_df(df, domain):
    if "xfact" in str(args.dataset):
        df = df[pd.to_datetime(df['claimDate'], errors='coerce').notna()]  # claimDate  for xfact
        df = df.dropna().sort_values(by='claimDate', na_position='first')  # claimDate  for xfact
        df['date'] = pd.to_datetime(df['claimDate']).dt.date  # claimDate  for xfact
    elif "yelp" in str(args.dataset):
        df['Year']=df['year']
    else:
        # df = df[pd.to_datetime(df['date'], errors='coerce').notna()]
        # df = df.dropna().sort_values(by='date', na_position='first')
        # df['date'] = pd.to_datetime(df['date']).dt.date
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.date
        df = df.dropna().sort_values(by='date', na_position='first')

    # if args.dataset == 'binarybragging':
    #     print(df['date'])
    #     df['Year'] = pd.to_datetime(df['date'].astype(str).str[:4]).dt.date
    #     print(df['Year'])
    #     # df['Year'] = df['date'].astype(str).str[:7]
    # else:
    #    df['Year'] = pd.DatetimeIndex(df['date']).year


    if "yelp" in str(args.dataset):
        df['Temporal Split'] = str(domain)
        stat_df = df[['Year', 'Temporal Split']]
    else:
        df['Year'] = pd.DatetimeIndex(df['date']).year
        df['Temporal Split'] = str(domain)
        stat_df = df[['Year', 'Temporal Split']]


    return stat_df

if args.plot_time_distribution:
############################# read data in ##################
    if "xfact" in str(args.dataset) or "factcheck" in str(args.dataset):
        # for xfact only, only read in adding, to make a dataset for full data
        full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
        full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
        full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'/data/dev.json')
        full_df_4 = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
        full_df_5 = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')
        full_df = pd.concat([full_df_1, full_df_2, full_df_3, full_df_4, full_df_5], ignore_index=False)
    # elif "factcheck" in str(args.dataset):
    #     full_df = pd.read_csv('./datasets/'+str(args.dataset)+'_full/data/'+str(args.dataset)+'_full.csv')
    else:
        full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/train.json') #'datasets/'+ str(args.dataset) +'_full/data/train.json')
        full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/test.json')
        full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/dev.json')
        full_df = pd.concat([full_df_1, full_df_2, full_df_3], ignore_index=False)


    indomain_train_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
    indomain_test_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
    ood1_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
    ood2_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')

    full = df2stat_df(full_df, 'Full-size')
    indomain_train = df2stat_df(indomain_train_df, 'SynD Train')
    indomain_test = df2stat_df(indomain_test_df, 'SynD Test')
    ood1 = df2stat_df(ood1_df, 'AsyD1 Test')
    ood2 = df2stat_df(ood2_df, 'AsyD2 Test')

    df = pd.concat([full, indomain_train, indomain_test, ood1, ood2]).reset_index(drop=True)

    #plt.style.use('ggplot')
    sns.violinplot(y=df['Year'], x=df['Temporal Split'], showmedians=True, showextrema=True, 
        palette="rocket",scale='width') #,gridsize=10

    plt.title(str(args.dataset).capitalize(), fontsize=18.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if args.dataset == 'yelp':
        yint = range(2005, 2023, 3)
        plt.yticks(yint,fontsize=12)
    # plt.ylabel("Percentage")
    #plt.xlabel("Full size", "InDomain Train", "InDomain Test", "OOD1 Test", "OOD2 Test")
    plt.tight_layout()
    plt.savefig('./TimeDist/'+str(args.dataset)+'_vio.png', bbox_inches = 'tight', dpi=250, format='png')
    plt.show()

# https://stackoverflow.com/questions/59346731/no-handles-with-labels-found-to-put-in-legend


######################## data statistic

from datetime import datetime

if args.combine_all_data_stat:
    df_list = []
    for i, task in enumerate(task_list):
        if i == 0:
            df = pd.read_csv('./saved_everything/'+ str(task) +'/dataset_stats.csv')
        else:
            temp_df = pd.read_csv('./saved_everything/'+ str(task) +'/dataset_stats.csv')
            df = pd.concat([df, temp_df])
    df.to_csv('./saved_everything/all_dataset_stats.csv')      


if args.save_data_stat:
    def df2stat_df(df, domain):
        print(df)
        if "xfact" in str(args.dataset):
            df = df[pd.to_datetime(df['claimDate'], errors='coerce').notna()] # claimDate  for xfact
            df = df.dropna().sort_values(by='claimDate', na_position='first') # claimDate  for xfact
            df['date'] = pd.to_datetime(df['claimDate']).dt.date              # claimDate  for xfact
        else:            
            df['date'] = pd.to_datetime(df['date'], errors = 'coerce', utc=True).dt.date
            df = df.dropna().sort_values(by='date', na_position='first') 
            print(df)
        
        label_dist = df['label'].value_counts().to_string()
        label_num = df['label'].nunique()
        print(df)

        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]

        print(df['date'])
        print('---', start_date)
        print('---', end_date)
        quartile = int(len(df) * 0.25)

        DATE = df['date'].tolist()
        Interquartile_start = DATE[quartile]
        Interquartile_Mid = DATE[int(quartile*2)]
        Interquartile_end = DATE[-quartile]

        duration = end_date - start_date
        inter_duration = Interquartile_end - Interquartile_start
        print('---duration ---')
        print(duration)
        if int(duration.days) <= 0:
            start_date = df['date'][len(df) - 1]
            end_date = df['date'][0]
            duration = end_date - start_date
        if int(inter_duration.days) <= 0:
            Interquartile_start = DATE[-quartile]
            Interquartile_end = DATE[quartile]
            inter_duration = Interquartile_end - Interquartile_start

        stat_df = pd.DataFrame({'Domain': [str(domain)], 'Label Num': [label_num], 'Label distribution': [label_dist], 'Interquartile-Oldest': [Interquartile_start],
                                'Median Date': [Interquartile_Mid], 'Interquartile-Newest': [Interquartile_end],
                                'Interquartile Time Span(Days)': [inter_duration.days],
                                'Start Date': [start_date], 'End Date': [end_date], 'Time Span(Days)': [duration.days],
                                'Data Num': [len(df)],
                                })
        
        stat_df['Interquartile Time Span(Days)'] = pd.to_numeric(stat_df['Interquartile Time Span(Days)'].astype(str).str.replace(r' days$', '', regex=True))
        stat_df['Time Span(Days)'] = pd.to_numeric(stat_df['Time Span(Days)'].astype(str).str.replace(r' days$', '', regex=True))

        stat_df['TimeDensity'] = stat_df['Data Num']/stat_df['Interquartile Time Span(Days)']
        stat_df['InterTimeDensity'] = stat_df['Data Num']*0.5/stat_df['Interquartile Time Span(Days)']
        stat_df['Data Num'] = pd.to_numeric(stat_df['Data Num'])

        print('done for :', str(domain))

        return stat_df


############################# read data in ##################
    if "xfact" in str(args.dataset):
        # for xfact only, only read in adding, to make a dataset for full data
        full_df_11 = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
        full_df_21 = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
        full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'/data/dev.json')
        full_df_4 = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
        full_df_5 = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')
        full_df = pd.concat([full_df_11, full_df_21, full_df_3, full_df_4, full_df_5], ignore_index=False)

        full_df_1 = pd.read_json('datasets/xfact_full/data/train.json')
        full_df_2 = pd.read_json('datasets/xfact_full/data/test.json')

    # elif "factcheck" in str(args.dataset):
    #     full_df = pd.read_csv('./datasets/'+str(args.dataset)+'_full/data/'+str(args.dataset)+'_full.json')
    else:
        full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/train.json')
        full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/test.json')
        full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/dev.json')
    
    full_df = pd.concat([full_df_1, full_df_2, full_df_3], ignore_index=False)

    indomain_train_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
    indomain_test_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
    ood1_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
    ood2_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')


    full = df2stat_df(full_df, 'Original (full size)')
    full_train = df2stat_df(full_df_1, 'Original Train')
    full_test = df2stat_df(full_df_2, 'Original Test')
    indomain_train = df2stat_df(indomain_train_df, 'In Domain Train')
    indomain_test = df2stat_df(indomain_test_df, 'InDomain')
    ood1 = df2stat_df(ood1_df, 'OOD1')
    ood2 = df2stat_df(ood2_df, 'OOD2')

    df = pd.concat([full, full_train, full_test, indomain_train, indomain_test, ood1, ood2])

    start_date = df.iloc[0]['Start Date']
    end_date = df.iloc[0]['End Date']
    Inter_start_date = df.iloc[0]['Interquartile-Oldest']
    Inter_end_date = df.iloc[0]['Interquartile-Newest']
    Median = df.iloc[0]['Median Date']

    TimeSpan = df.iloc[0]['Interquartile Time Span(Days)']
    print('==============')
    print(TimeSpan)
    InterTimeSpan = df.iloc[0]['Interquartile Time Span(Days)']
    
    print(abs(df['Start Date']-start_date))

    df['TimeDiff'] = abs(df['Start Date']-start_date) + abs(df['End Date']-end_date) + abs(df['Interquartile-Oldest']-Inter_start_date) + abs(df['Interquartile-Newest']-Inter_end_date) + abs(df['Median Date']-Median)
    df['TimeDiff'] = pd.to_numeric(df['TimeDiff'].astype(str).str.replace(r' days$', '', regex=True))/(TimeSpan*0.5+InterTimeSpan*0.5)
    print(df['TimeDiff'])

    df['Task'] = str(args.dataset)

    df = df[['Task', 'Label Num', 'Domain', 'Start Date', 'End Date', 'Time Span(Days)', 'Median Date',
            'Interquartile Time Span(Days)', 'Data Num']]

    df.to_csv('./saved_everything/'+ str(args.dataset) +'/dataset_stats.csv')



######################## 1. bert predictive resultes -- on In domain / ood1 / ood2
if args.save_for_bert:
    Full_data = pd.read_json('./models/' + str(args.dataset) + '_full/bert_predictive_performances.json')
    InDomain = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances.json')
    path = os.path.join('./models/', str(args.dataset),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
    OOD1 = pd.read_json(path)
    path = os.path.join('./models/', str(args.dataset),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
    OOD2 = pd.read_json(path)

    if args.get_all_seeds_for_predictive:
        pass
    else:
        Full_data = Full_data[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        InDomain = InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        OOD1 = OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        OOD2 = OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]

    OOD12 = (OOD1 + OOD2)/2
    Full_data['Domain'] = 'Full size'
    InDomain['Domain'] = 'SynD'
    OOD1['Domain'] = 'AsyD1'
    OOD2['Domain'] = 'AsyD2'
    #OOD12['Domain'] = 'AsyD1+2'
    

    result = pd.concat([Full_data, InDomain, OOD1, OOD2], ignore_index=False, axis=1).T
    result.to_csv('saved_everything/' + str(args.dataset) + '/bert_predictive.csv')
####################################################################################











######################################## 2. faithful of different measures & different attributes rationales for both top / contigious -- on In domain / ood1 / ood2 #########################

def json2df(df, domain):
    df.rename(columns={"": "Task"})
    list_of_list = []
    # df.columns = ['lime', 'gradients', 'scaled attention', 'random', 'deeplift', 'attention']
    for col, col_attribute_name in enumerate(df.columns): # the range length equals to the number of attributes, if remove ig
        rationales_sufficiency = df.iloc[0, col].get('mean')
        rationales_comprehensiveness = df.iloc[1, col].get('mean')
        rationales_AOPCsufficiency = df.iloc[2, col].get('mean')
        rationales_AOPCcomprehensiveness = df.iloc[3, col].get('mean')

        four_eval_metrics = [rationales_sufficiency, rationales_comprehensiveness,
                             rationales_AOPCsufficiency, rationales_AOPCcomprehensiveness]

        list_of_list.append(four_eval_metrics)
        # print(four_eval_metrics)

    df_tf = pd.DataFrame.from_records(list_of_list).transpose()
    df_tf.columns = df.columns  # ['random','scaled_attention','attention','ig','lime','gradients','deeplift']
    # df_tf = df_tf[['random','scaled attention','attention','deeplift','gradients','lime']]

    df_tf['Rationales_metrics'] = ['Sufficiency', 'Comprehensiveness', 'AOPC_sufficiency', 'AOPC_comprehensiveness']
    df_tf['Domain'] = str(domain)
    df_tf = df_tf.set_index('Rationales_metrics')
    return df_tf



if args.save_posthoc:
    df_list = []
    for thresh in ['topk', 'contigious']:

        for fname in os.listdir('posthoc_results/' + str(args.dataset)):
            if str(thresh) in fname and 'description.json' in fname:
                if 'ood1' in fname:
                    ood1_path = os.path.join('posthoc_results', str(args.dataset), fname)
                elif 'ood2' in fname:
                    ood2_path = os.path.join('posthoc_results', str(args.dataset), fname)
                else:
                    indomain_path = os.path.join('posthoc_results', str(args.dataset), fname)
        
        for fname in os.listdir('posthoc_results/' + str(args.dataset) + '_full/'):
            if 'OOD' not in fname and str(thresh) in fname and 'description.json' in fname:
                full_path = os.path.join('posthoc_results', str(args.dataset)+'_full', fname)
                
                

        full = pd.read_json(full_path)
        full_df = json2df(full, 'Full size')
        indomain = pd.read_json(indomain_path)
        df = json2df(indomain, 'SynD')
        OOD1 = pd.read_json(ood1_path)
        df1 = json2df(OOD1, 'AsyD1')
        OOD2 = pd.read_json(ood2_path)
        df2 = json2df(OOD2, 'AsyD2')

        final = pd.concat([full_df, df, df1, df2], ignore_index=False)
        #final = pd.concat([df, df1, df2], ignore_index=False)
        final['thresholder'] = str(thresh)
        df_list.append(final)


    posthoc_faithfulness = pd.concat([df_list[0], df_list[1]], ignore_index=False)
    print(posthoc_faithfulness)
    # posthoc_faithfulness = pd.concat([seed_list[0],seed_list[1],seed_list[2],seed_list[3],seed_list[4]], ignore_index=False)
    posthoc_faithfulness.to_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness.csv')

    if args.save_posthoc_for_analysis:
        print(df_list[0])

        topk = df_list[0][['Domain', 'random', 
                            'scaled attention','attention','deeplift','gradients','lime','ig','deepliftshap',
                            'gradientshap']]
        topk['scaled attention'] = topk['scaled attention']/topk['random']
        topk['attention'] = topk['attention']/topk['random']
        topk['deeplift'] = topk['deeplift']/topk['random']
        topk['gradients'] = topk['gradients']/topk['random']
        topk['lime'] = topk['lime']/topk['random']
        topk['ig'] = topk['ig']/topk['random']
        topk['deepliftshap'] = topk['deepliftshap']/topk['random']
        topk['gradientshap'] = topk['gradientshap']/topk['random']
        topk = topk[['Domain','scaled attention','attention','deeplift','gradients','lime','ig','deepliftshap','gradientshap']]

        AOPC_sufficiency = topk.loc[['AOPC_sufficiency']].set_index('Domain')
        OOD12 = (AOPC_sufficiency.loc['AsyD1'] + AOPC_sufficiency.loc['AsyD2'])/2
        OOD12.name = 'AsyD1+2'
        AOPC_sufficiency = AOPC_sufficiency.append([OOD12])

        AOPC_comprehensiveness = topk.loc[['AOPC_comprehensiveness']].set_index('Domain')
        OOD12 = (AOPC_comprehensiveness.loc['AsyD1'] + AOPC_comprehensiveness.loc['AsyD2'])/2
        OOD12.name = 'AsyD1+2'
        AOPC_comprehensiveness = AOPC_comprehensiveness.append([OOD12])

        final = pd.concat([AOPC_sufficiency, AOPC_comprehensiveness], axis=1)
        final.to_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness_overleaf.csv')
        pd.set_option('display.max_columns', None)
        pd.options.display.float_format = "{:,.2f}".format
        print(final)

#############################################################################################################################################


    

import plotly.graph_objects as go
import plotly.offline as pyo
from selenium import webdriver
from plotly.subplots import make_subplots

if args.plot_radar:
    # args.save_posthoc == True
    # args.save_posthoc_for_analysis == True
    df = pd.read_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness_overleaf.csv', index_col=0)
    APOC_sufficiency = df.iloc[:, 0:8]
    AOPC_comprehensiveness = df.iloc[:, -8:]
    print(APOC_sufficiency)
    print(AOPC_comprehensiveness)
    categories = APOC_sufficiency.columns
    print(categories)
    categories = [*categories, categories[0]]

    Full_sufficiency = APOC_sufficiency.iloc[0, :].to_list()
    SynD_sufficiency = APOC_sufficiency.iloc[1, :].to_list()
    AsyD1_sufficiency = APOC_sufficiency.iloc[2, :].to_list()
    AsyD2_sufficiency = APOC_sufficiency.iloc[3, :].to_list()
    #AsyD1_2_sufficiency = APOC_sufficiency.iloc[4, :].to_list()
    Full_sufficiency = [*Full_sufficiency, Full_sufficiency[0]]
    SynD_sufficiency = [*SynD_sufficiency, SynD_sufficiency[0]]
    AsyD1_sufficiency = [*AsyD1_sufficiency, AsyD1_sufficiency[0]]
    AsyD2_sufficiency = [*AsyD2_sufficiency, AsyD2_sufficiency[0]]
    #AsyD1_2 = [*AsyD1_2, AsyD1_2[0]]


    Full_comprehensiveness = AOPC_comprehensiveness.iloc[0, :].to_list()
    SynD_comprehensiveness = AOPC_comprehensiveness.iloc[1, :].to_list()
    AsyD1_comprehensiveness = AOPC_comprehensiveness.iloc[2, :].to_list()
    AsyD2_comprehensiveness = AOPC_comprehensiveness.iloc[3, :].to_list()
    #AsyD1_2_sufficiency = APOC_sufficiency.iloc[4, :].to_list()
    Full_comprehensiveness = [*Full_comprehensiveness, Full_comprehensiveness[0]]
    SynD_comprehensiveness = [*SynD_comprehensiveness, SynD_comprehensiveness[0]]
    AsyD1_comprehensiveness = [*AsyD1_comprehensiveness, AsyD1_comprehensiveness[0]]
    AsyD2_comprehensiveness = [*AsyD2_comprehensiveness, AsyD2_comprehensiveness[0]]


    fig = go.Figure(data=[go.Scatterpolar(r=Full_sufficiency, theta=categories, name='Full'),
                        go.Scatterpolar(r=SynD_sufficiency, theta=categories, name='SynD'),
                        go.Scatterpolar(r=AsyD1_sufficiency, theta=categories, name='AsyD1'),
                        go.Scatterpolar(r=AsyD2_sufficiency, theta=categories, name='AsyD2'),
                        ],
    layout=go.Layout(
        title=go.layout.Title(text='APOC Sufficiency of Different Time Span'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True))
    fig.update_layout(
        font_family="Courier New",
        font_color="Black",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="Black",
        legend_title="Time",
        font=dict(family="Courier New, monospace",
                size=32,
                color="black", #RebeccaPurple
                ),
        legend=dict(yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.15,
            ),
        title={
        # 'text': "Plot Title",
        'y':0.9999,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_xaxes(title_font_family="Arial")
    fig.show()
    #pyo.plot(fig)



    fig = go.Figure(data=[go.Scatterpolar(r=Full_comprehensiveness, theta=categories, name='Full'),
                        go.Scatterpolar(r=SynD_comprehensiveness, theta=categories, name='SynD'),
                        go.Scatterpolar(r=AsyD1_comprehensiveness, theta=categories, name='AsyD1'),
                        go.Scatterpolar(r=AsyD2_comprehensiveness, theta=categories, name='AsyD2'),
                        ],
    layout=go.Layout(
        title=go.layout.Title(text='APOC Comprehensiveness of Different Time Span'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True))
    fig.update_layout(
        font_family="Courier New",
        font_color="Black",
        title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="Black",
        legend_title="Time",
        font=dict(family="Courier New, monospace",
                size=39,
                color="black", #RebeccaPurple
                ),
        legend=dict(yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.15,
            ),
        title={
            # 'text': "Plot Title",
            'y':0.9999,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
            xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 0.5, 1, 1.5, 2, 2.5],
        #ticktext = ['One', 'Three', 'Five', 'Seven', 'Nine', 'Eleven']
    )
        )
    fig.update_xaxes(title_font_family="Arial")
    fig.show()



    



########################### 3. FRESH results
if args.save_for_fresh:
    select_columns = ['mean-acc','std-acc','mean-f1','std-f1','mean-ece','std-ece']
    thresh_hold_list = []
    for threshold in ['topk', 'contigious']: #
        attribute_list = []
        for attribute_name in ["attention", "gradients", "lime", "deeplift", "scaled_attention"]:
            path = os.path.join('FRESH_classifiers/', str(args.dataset), str(threshold),
                                str(attribute_name) + '_bert_predictive_performances.json')
            print(path)
            fresh_InDomain = pd.read_json(path)
            fresh_InDomain = fresh_InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
            fresh_InDomain['Domain'] = 'InDomain'
            path1 = './FRESH_classifiers/' + str(args.dataset) + '/topk/attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json'
            fresh_OOD1 = pd.read_json(path1)
            fresh_OOD1 = fresh_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
            fresh_OOD1['Domain'] = 'OOD1'

            path2 = './FRESH_classifiers/' + str(args.dataset) + '/topk/attention_bert_predictive_performances-OOD-' + str(
                args.dataset) + '_ood2.json'
            fresh_OOD2 = pd.read_json(path2)
            fresh_OOD2 = fresh_OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
            fresh_OOD2['Domain'] = 'OOD2'

            attribute_df = pd.concat([fresh_InDomain, fresh_OOD1, fresh_OOD2], axis=1, ignore_index=False).T.reset_index()[
                ['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece', 'Domain']]
            attribute_df['attribute_name'] = str(attribute_name)
            attribute_list.append(attribute_df)

        attribute_results = pd.concat([attribute_list[0], attribute_list[1], attribute_list[2], attribute_list[3], attribute_list[4]], ignore_index=False)
        attribute_results['threshold'] = str(threshold)
        thresh_hold_list.append(attribute_results)

    fresh_final_result = pd.concat([thresh_hold_list[0], thresh_hold_list[1]], ignore_index=False)
    fresh_final_result.to_csv('saved_everything/' + str(args.dataset) + '/fresh_predictive_results.csv')





####################################  KUMA AND LSTM ############################################################

if args.save_all_selective_rationales:
    args.save_for_kuma_lstm == True

if args.save_for_kuma_lstm:

    ## get KUMA of FULL / IN D / OOD1 / OOD2
    kuma_FullData = pd.read_json('./kuma_model/' + str(args.dataset) + '_full/kuma-bert_predictive_performances.json')
    kuma_InDomain = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances.json')
    kuma_OOD1 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(
        args.dataset) + '_ood1.json')
    kuma_OOD2 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(
        args.dataset) + '_ood2.json')

    LSTM_FullData = pd.read_json(
        './LSTM_model/' + str(args.dataset) + '_full/full_lstm-bert_predictive_performances.json')
    LSTM_InDomain = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances.json')
    LSTM_OOD1 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(
        args.dataset) + '_ood1.json')
    LSTM_OOD2 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(
        args.dataset) + '_ood2.json')

    if args.get_all_seeds_for_predictive:
        pass
    else:
        kuma_FullData = kuma_FullData[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        kuma_InDomain = kuma_InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        kuma_OOD1 = kuma_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        kuma_OOD2 = kuma_OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]

        LSTM_FullData = LSTM_FullData[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        LSTM_InDomain = LSTM_InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        LSTM_OOD1 = LSTM_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
        LSTM_OOD2 = LSTM_OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]

    kuma_FullData['Domain'] = 'Full'
    kuma_InDomain['Domain'] = 'SynD'
    kuma_OOD1['Domain'] = 'AsyD1'
    kuma_OOD2['Domain'] = 'AsyD2'

    LSTM_FullData['Domain'] = 'Full'
    LSTM_InDomain['Domain'] = 'SynD'
    LSTM_OOD1['Domain'] = 'AsyD1'
    LSTM_OOD2['Domain'] = 'AsyD2'

    kuma_result = pd.concat([kuma_FullData, kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False, axis=1).T
    kuma_result['Model'] = 'Kuma'
    print(kuma_result)

    LSTM_result = pd.concat([LSTM_FullData, LSTM_InDomain, LSTM_OOD1, LSTM_OOD2], ignore_index=False, axis=1).T
    LSTM_result['Model'] = 'LSTM'

    final = pd.concat([kuma_result, LSTM_result], ignore_index=False)
    final.to_csv('saved_everything/' + str(args.dataset) + '/KUMA_LSTM_predictive_results.csv')



task_list = ['agnews', 'xfact', 'factcheck', 'AmazDigiMu', 'AmazPantry', 'yelp']

# if args.save_all_selective_rationales:

#     ## get KUMA of FULL / IN D / OOD1 / OOD2
#     kuma_FullData = pd.read_json('./kuma_model/' + str(args.dataset) + '_full/kuma-bert_predictive_performances.json')
#     kuma_InDomain = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances.json')
#     kuma_OOD1 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(
#         args.dataset) + '_ood1.json')
#     kuma_OOD2 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(
#         args.dataset) + '_ood2.json')

#     LSTM_FullData = pd.read_json(
#         './LSTM_model/' + str(args.dataset) + '_full/full_lstm-bert_predictive_performances.json')
#     LSTM_InDomain = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances.json')
#     LSTM_OOD1 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(
#         args.dataset) + '_ood1.json')
#     LSTM_OOD2 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(
#         args.dataset) + '_ood2.json')

#     kuma_FullData = kuma_FullData[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
#     kuma_InDomain = kuma_InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
#     kuma_OOD1 = kuma_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
#     kuma_OOD2 = kuma_OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]

#     LSTM_FullData = LSTM_FullData[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
#     LSTM_InDomain = LSTM_InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
#     LSTM_OOD1 = LSTM_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]
#     LSTM_OOD2 = LSTM_OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[0]

#     kuma_FullData['Domain'] = 'Full'
#     kuma_InDomain['Domain'] = 'SynD'
#     kuma_OOD1['Domain'] = 'AsyD1'
#     kuma_OOD2['Domain'] = 'AsyD2'

#     LSTM_FullData['Domain'] = 'Full'
#     LSTM_InDomain['Domain'] = 'SynD'
#     LSTM_OOD1['Domain'] = 'AsyD1'
#     LSTM_OOD2['Domain'] = 'AsyD2'

#     kuma_result = pd.concat([kuma_FullData, kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False, axis=1).T
#     LSTM_result = pd.concat([LSTM_FullData, LSTM_InDomain, LSTM_OOD1, LSTM_OOD2], ignore_index=False, axis=1).T

#     LSTM_result = LSTM_result[['mean-f1', 'domain']]
#     LSTM_result = LSTM_result.rename({'mean-f1': 'LSTM'})
#     kuma_result = kuma_result[['mean-f1', 'domain']]
#     kuma_result = kuma_result.rename({'mean-f1': 'KUMA'})
#     BERT_results = pd.read_csv('saved_everything/' + str(args.dataset) + '/bert_predictive.csv')[['mean-f1', 'Domain']]
#     BERT_results = BERT_results.rename({'mean-f1': 'BERT'})




if args.predictive_and_posthoc:
    args.save_posthoc == True
    args.save_for_bert == True

    bert = pd.read_csv('saved_everything/' + str(args.dataset) + '/bert_predictive.csv')[['mean-f1', 'Domain']]
    posthoc = pd.read_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness.csv')

    merge = pd.merge(posthoc, bert, on = 'Domain')
    #merge = merge.rename(columns={merge.columns[1]: 'Task'},inplace=True)
    if 'Amaz' in str(args.dataset):
        merge['Task'] = str(args.dataset)
    else:
        merge['Task'] = str(args.dataset).capitalize()
    print(merge)
    merge.to_csv('saved_everything/' + str(args.dataset) + '/posthoc_and_predictive.csv')