# 1. bert predictive resultes -- on In domain / ood1 / ood2
# 2. faithful: for both top / contigious -- on In domain / ood1 / ood2
# 3. FRESH results
# 4. kuma results (another script)
# 5. domain similarity between:  In domain / ood1 / ood2
# 6. rationale similarity between:  In domain / ood1 / ood2

# 7. datasets metadata: train/test/ size, time span, label distribution

import pandas as pd
import json
import csv
import config.cfg
import os
import argparse
import fnmatch



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
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)



######################## data statistic

from datetime import datetime


def df2stat_df(df, domain):
    df = df[pd.to_datetime(df['date'], errors='coerce').notna()] # claimDate  for xfact
    df = df.dropna().sort_values(by='date', na_position='first') # claimDate  for xfact
    df['date'] = pd.to_datetime(df['date']).dt.date              # claimDate  for xfact
    label_dist = df['label'].value_counts().to_string()


    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]

    print(df['date'])
    print('---', start_date)
    print('---', end_date)
    inter_quartile = df.date.quantile([0.25, 0.5, 0.75])
    Interquartile_start = inter_quartile.values[0]
    Interquartile_Mid = inter_quartile.values[1]
    Interquartile_end = inter_quartile.values[2]

    duration = end_date - start_date
    inter_duration = Interquartile_end - Interquartile_start
    print('---duration ---')
    print(duration)
    if int(duration.days) <= 0:
        start_date = df['date'][len(df) - 1]
        end_date = df['date'][0]
        duration = end_date - start_date

    if int(inter_duration.days) <= 0:
        Interquartile_start = df['date'][len(df) - 1]
        Interquartile_end = df['date'][0]
        duration = end_date - start_date

    stat_df = pd.DataFrame({'Domain': [str(domain)], 'Label distribution': [label_dist], 'Interquartile - Oldest': [Interquartile_start],
                            'Median': [Interquartile_Mid], 'Interquartile - Newest': [Interquartile_end],
                            'Interquartile Time Span in Days': [inter_duration],
                            'Oldest Date': [start_date], 'Newest Date': [end_date], 'Time Span in Days': [duration],
                            'Data Num': [len(df)],
                            })
    print('done for :', str(domain))

    return stat_df

############################# read data in ##################
# full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/train.json')
# full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/test.json')
# full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'_full/data/dev.json')
# full_df = pd.concat([full_df_1, full_df_2, full_df_3], ignore_index=False)

# for factcheck
full_df = pd.read_csv('./datasets/factcheck_full/data/factcheck_full.csv')

# for xfact only, only read in adding, to make a dataset for full data
# full_df_1 = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
# full_df_2 = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
# full_df_3 = pd.read_json('datasets/'+ str(args.dataset) +'/data/dev.json')
# full_df_4 = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
# full_df_5 = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')
# full_df = pd.concat([full_df_1, full_df_2, full_df_3, full_df_4, full_df_5], ignore_index=False)


#################################################################

indomain_train_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
indomain_test_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
ood1_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
ood2_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')

full = df2stat_df(full_df, 'Full')
indomain_train = df2stat_df(indomain_train_df, 'In Domain Train')
indomain_test = df2stat_df(indomain_test_df, 'In Domain Test')
ood1 = df2stat_df(ood1_df, 'OOD1')
ood2 = df2stat_df(ood2_df, 'OOD2')

df = pd.concat([full, indomain_train, indomain_test, ood1, ood2])
df.to_csv('./saved_everything/'+ str(args.dataset) +'/dataset_stats.csv')





######################## 1. bert predictive resultes -- on In domain / ood1 / ood2

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


Full_data['domain'] = 'Full size'
InDomain['domain'] = 'InDomain'
OOD1['domain'] = 'OOD1'
OOD2['domain'] = 'OOD2'
result = pd.concat([Full_data, InDomain, OOD1, OOD2], ignore_index=False, axis=1).T
result.to_csv('saved_everything/' + str(args.dataset) + '/bert_predictive_on_fulltext.csv')
####################################################################################











######################################## 2. faithful of different measures & different attributes rationales for both top / contigious -- on In domain / ood1 / ood2 #########################

def json2df(df, domain):
    df.rename(columns={"": "Task"})
    list_of_list = []

    for col in range(0, len(df.columns)): # the range length equals to the number of attributes, if remove ig
        rationales_sufficiency = df.iloc[0, col].get('mean')
        rationales_comprehensiveness = df.iloc[1, col].get('mean')
        rationales_AOPCsufficiency = df.iloc[2, col].get('mean')
        rationales_AOPCcomprehensiveness = df.iloc[3, col].get('mean')

        four_eval_metrics = [rationales_sufficiency, rationales_comprehensiveness,
                             rationales_AOPCsufficiency, rationales_AOPCcomprehensiveness]

        list_of_list.append(four_eval_metrics)

    df_tf = pd.DataFrame.from_records(list_of_list).transpose()
    # print(df_tf)
    df_tf.columns = df.columns  # ['random','scaled_attention','attention','ig','lime','gradients','deeplift']

    df_tf['Rationales_metrics'] = ['Sufficiency', 'Comprehensiveness', 'AOPC_sufficiency', 'AOPC_comprehensiveness']
    df_tf['Domain'] = str(domain)
    df_tf = df_tf.set_index('Rationales_metrics')
    return df_tf




#
# seed_list = []
# for seed in [10]: #[5,10,15,20,25]:
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

    json = pd.read_json(indomain_path)
    df = json2df(json, 'InDomain')
    OOD1 = pd.read_json(ood1_path)
    df1 = json2df(OOD1, 'OOD1')
    OOD2 = pd.read_json(ood2_path)
    df2 = json2df(OOD2, 'OOD2')

    final = pd.concat([df, df1, df2], ignore_index=False)
    final['thresholder'] = str(thresh)
    df_list.append(final)

posthoc_faithfulness = pd.concat([df_list[0], df_list[1]], ignore_index=False)
    # seed_n['seed'] = seed
    # seed_list.append(seed_n)

# posthoc_faithfulness = pd.concat([seed_list[0],seed_list[1],seed_list[2],seed_list[3],seed_list[4]], ignore_index=False)
posthoc_faithfulness.to_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness.csv')
exit()
#############################################################################################################################################


# 3. FRESH results
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
        fresh_InDomain['domain'] = 'InDomain'
        path1 = './FRESH_classifiers/' + str(args.dataset) + '/topk/attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json'
        fresh_OOD1 = pd.read_json(path1)
        fresh_OOD1 = fresh_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
        fresh_OOD1['domain'] = 'OOD1'

        path2 = './FRESH_classifiers/' + str(args.dataset) + '/topk/attention_bert_predictive_performances-OOD-' + str(
            args.dataset) + '_ood2.json'
        fresh_OOD2 = pd.read_json(path2)
        fresh_OOD2 = fresh_OOD2[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
        fresh_OOD2['domain'] = 'OOD2'

        attribute_df = pd.concat([fresh_InDomain, fresh_OOD1, fresh_OOD2], axis=1, ignore_index=False).T.reset_index()[
            ['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece', 'domain']]
        attribute_df['attribute_name'] = str(attribute_name)
        attribute_list.append(attribute_df)

    attribute_results = pd.concat([attribute_list[0], attribute_list[1], attribute_list[2], attribute_list[3], attribute_list[4]], ignore_index=False)
    attribute_results['threshold'] = str(threshold)
    thresh_hold_list.append(attribute_results)

fresh_final_result = pd.concat([thresh_hold_list[0], thresh_hold_list[1]], ignore_index=False)
fresh_final_result.to_csv('saved_everything/' + str(args.dataset) + '/fresh_predictive_results.csv')





################################################################################################

