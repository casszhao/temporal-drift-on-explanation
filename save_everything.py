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
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)



######################## plot time distribution

from datetime import datetime

if args.plot_time_distribution:
    def df2stat_df(df, domain):
        if "xfact" in str(args.dataset):
            df = df[pd.to_datetime(df['claimDate'], errors='coerce').notna()] # claimDate  for xfact
            df = df.dropna().sort_values(by='claimDate', na_position='first') # claimDate  for xfact
            df['date'] = pd.to_datetime(df['claimDate']).dt.date              # claimDate  for xfact
        else:
            df = df[pd.to_datetime(df['date'], errors='coerce').notna()] 
            df = df.dropna().sort_values(by='date', na_position='first') 
            df['date'] = pd.to_datetime(df['date']).dt.date  

        if args.dataset == 'binarybragging':
            df['Year'] = pd.to_datetime(df['date'].astype(str).str[:7])
            #df['Year'] = df['date'].astype(str).str[:7]
        else:       
            df['Year'] = pd.DatetimeIndex(df['date']).year
        df['Domain'] = str(domain)

        stat_df = df[['Year', 'Domain']]
        
        return stat_df


############################# read data in ##################
    if "xfact" in str(args.dataset):
        # for xfact only, only read in adding, to make a dataset for full data
        full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
        full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
        full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'/data/dev.json')
        full_df_4 = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
        full_df_5 = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')
        full_df = pd.concat([full_df_1, full_df_2, full_df_3, full_df_4, full_df_5], ignore_index=False)

    elif "factcheck" in str(args.dataset):
        full_df = pd.read_csv('./datasets/'+str(args.dataset)+'_full/data/'+str(args.dataset)+'_full.csv')
    else:
        full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/train.json') #'datasets/'+ str(args.dataset) +'_full/data/train.json')
        full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/test.json')
        full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/dev.json')
        full_df = pd.concat([full_df_1, full_df_2, full_df_3], ignore_index=False)


    indomain_train_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
    indomain_test_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
    ood1_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
    ood2_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')

    full = df2stat_df(full_df, 'Full')
    indomain_train = df2stat_df(indomain_train_df, 'InDomain Train')
    indomain_test = df2stat_df(indomain_test_df, 'InDomain Test')
    ood1 = df2stat_df(ood1_df, 'OOD1')
    ood2 = df2stat_df(ood2_df, 'OOD2')

    df = pd.concat([full, indomain_train, indomain_test, ood1, ood2]).reset_index(drop=True)
    # pd.to_numeric(df['Year'], downcast='integer')
    print(df)
    # penguins = sns.load_dataset("penguins")
    # print(penguins)
    # sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")
    # plt.show()
    if args.dataset == 'binarybragging':
        sns.displot(x=df['Year'],hue=df['Domain'], kind="kde") 
    else:
        sns.displot(x=df['Year'],hue=df['Domain'], kind="kde") 
    plt.title(str(args.dataset), fontsize=20)
    plt.ylabel("Percentage")
    plt.legend(bbox_to_anchor=(1, 1, 0.28, 0.28), loc='best', borderaxespad=1)
    plt.tight_layout()
    plt.savefig('./TimeDist/'+str(args.dataset)+'.png', bbox_inches = 'tight', dpi=200, format='png')
    plt.show()

# https://stackoverflow.com/questions/59346731/no-handles-with-labels-found-to-put-in-legend


######################## data statistic

from datetime import datetime

if args.save_data_stat:
    def df2stat_df(df, domain):
        if "xfact" in str(args.dataset):
            df = df[pd.to_datetime(df['claimDate'], errors='coerce').notna()] # claimDate  for xfact
            df = df.dropna().sort_values(by='claimDate', na_position='first') # claimDate  for xfact
            df['date'] = pd.to_datetime(df['claimDate']).dt.date              # claimDate  for xfact
        else:            
            df['date'] = pd.to_datetime(df['date'], errors = 'coerce', utc=True).dt.date
            df = df.dropna().sort_values(by='date', na_position='first') 
        
        label_dist = df['label'].value_counts().to_string()

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
            Interquartile_start = df['date'][int(quartile*3)]
            Interquartile_end = df['date'][quartile]
            inter_duration = Interquartile_end - Interquartile_start

        stat_df = pd.DataFrame({'Domain': [str(domain)], 'Label distribution': [label_dist], 'Interquartile-Oldest': [Interquartile_start],
                                'Median': [Interquartile_Mid], 'Interquartile-Newest': [Interquartile_end],
                                'InterTimeSpan(D)': [inter_duration.days],
                                'Oldest Date': [start_date], 'Newest Date': [end_date], 'TimeSpan(D)': [duration.days],
                                'Data Num': [len(df)],
                                })
        
        stat_df['InterTimeSpan(D)'] = pd.to_numeric(stat_df['InterTimeSpan(D)'].astype(str).str.replace(r' days$', '', regex=True))
        stat_df['TimeSpan(D)'] = pd.to_numeric(stat_df['TimeSpan(D)'].astype(str).str.replace(r' days$', '', regex=True))

        stat_df['TimeDensity'] = stat_df['Data Num']/stat_df['TimeSpan(D)']
        stat_df['InterTimeDensity'] = stat_df['Data Num']*0.5/stat_df['InterTimeSpan(D)']
        stat_df['Data Num'] = pd.to_numeric(stat_df['Data Num'])

        print('done for :', str(domain))

        return stat_df


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


    full = df2stat_df(full_df, 'Full')
    indomain_train = df2stat_df(indomain_train_df, 'In Domain Train')
    indomain_test = df2stat_df(indomain_test_df, 'InDomain')
    ood1 = df2stat_df(ood1_df, 'OOD1')
    ood2 = df2stat_df(ood2_df, 'OOD2')

    df = pd.concat([full, indomain_train, indomain_test, ood1, ood2])

    start_date = df.iloc[0]['Oldest Date']
    end_date = df.iloc[0]['Newest Date']
    Inter_start_date = df.iloc[0]['Interquartile-Oldest']
    Inter_end_date = df.iloc[0]['Interquartile-Newest']
    Median = df.iloc[0]['Median']

    TimeSpan = df.iloc[0]['TimeSpan(D)']
    print('==============')
    print(TimeSpan)
    InterTimeSpan = df.iloc[0]['InterTimeSpan(D)']
    
    print(abs(df['Oldest Date']-start_date))

    df['TimeDiff'] = abs(df['Oldest Date']-start_date) + abs(df['Newest Date']-end_date) + abs(df['Interquartile-Oldest']-Inter_start_date) + abs(df['Interquartile-Newest']-Inter_end_date) + abs(df['Median']-Median)
    df['TimeDiff'] = pd.to_numeric(df['TimeDiff'].astype(str).str.replace(r' days$', '', regex=True))/(TimeSpan*0.5+InterTimeSpan*0.5)
    print(df['TimeDiff'])

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


    Full_data['Domain'] = 'Full size'
    InDomain['Domain'] = 'InDomain'
    OOD1['Domain'] = 'OOD1'
    OOD2['Domain'] = 'OOD2'
    result = pd.concat([Full_data, InDomain, OOD1, OOD2], ignore_index=False, axis=1).T
    result.to_csv('saved_everything/' + str(args.dataset) + '/bert_predictive_on_fulltext.csv')
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
    df_tf = df_tf[['random','scaled attention','attention','deeplift','gradients','lime']]

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
                # print('full: ', full_path)

        full = pd.read_json(full_path)
        full_df = json2df(full, 'Full')
        json = pd.read_json(indomain_path)
        df = json2df(json, 'InDomain')
        OOD1 = pd.read_json(ood1_path)
        df1 = json2df(OOD1, 'OOD1')
        OOD2 = pd.read_json(ood2_path)
        df2 = json2df(OOD2, 'OOD2')

        final = pd.concat([full_df, df, df1, df2], ignore_index=False)
        final['thresholder'] = str(thresh)
        df_list.append(final)

    posthoc_faithfulness = pd.concat([df_list[0], df_list[1]], ignore_index=False)
        # seed_n['seed'] = seed
        # seed_list.append(seed_n)

    # posthoc_faithfulness = pd.concat([seed_list[0],seed_list[1],seed_list[2],seed_list[3],seed_list[4]], ignore_index=False)
    posthoc_faithfulness.to_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness.csv')

#############################################################################################################################################


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

    kuma_FullData['Domain'] = 'Full data'
    kuma_InDomain['Domain'] = 'InDomain'
    kuma_OOD1['Domain'] = 'OOD1'
    kuma_OOD2['Domain'] = 'OOD2'

    LSTM_FullData['Domain'] = 'Full data'
    LSTM_InDomain['Domain'] = 'InDomain'
    LSTM_OOD1['Domain'] = 'OOD1'
    LSTM_OOD2['Domain'] = 'OOD2'

    kuma_result = pd.concat([kuma_FullData, kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False, axis=1).T
    kuma_result['Model'] = 'Kuma'
    print(kuma_result)

    LSTM_result = pd.concat([LSTM_FullData, LSTM_InDomain, LSTM_OOD1, LSTM_OOD2], ignore_index=False, axis=1).T
    LSTM_result['Model'] = 'LSTM'

    final = pd.concat([kuma_result, LSTM_result], ignore_index=False)
    final.to_csv('saved_everything/' + str(args.dataset) + '/KUMA_LSTM_predictive_results.csv')
