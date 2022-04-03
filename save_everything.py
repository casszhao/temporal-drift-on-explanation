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
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)



######################## data statistic

from datetime import datetime



def df2stat_df(df, domain):
    df = df[pd.to_datetime(df['date'], errors='coerce').notna()]
    df = df.dropna().sort_values(by='date', na_position='first')
    df['date'] = pd.to_datetime(df['date'])
    start_date1 = df['date'][0]
    end_date1 = df['date'][len(df)-1]
    # print(start_date1)
    # print(end_date1)
    duration1 = end_date1-start_date1
    if int(duration1.days) <= 0:
        start_date1 = df['date'][len(df)-1]
        end_date1 = df['date'][0]
        duration1 = end_date1 - start_date1
    temporal_density1 = int(duration1.days)/len(df)
    stat_df = pd.DataFrame({'Domain': [str(domain)], 'Oldest Date': [start_date1], 'Newest Date': [end_date1], 'Duration in Days': [duration1],
               'Data Num': [len(df)], 'Temporal Density': [temporal_density1]})
    return stat_df


indomain_train_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/train.json')
indomain_test_df = pd.read_json('datasets/'+ str(args.dataset) +'/data/test.json')
ood1_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood1/data/test.json')
ood2_df = pd.read_json('datasets/'+ str(args.dataset) +'_ood2/data/test.json')

indomain_train = df2stat_df(indomain_train_df, 'In Domain Train')
indomain_test = df2stat_df(indomain_test_df, 'In Domain Test')
ood1 = df2stat_df(ood1_df, 'OOD1')
ood2 = df2stat_df(ood2_df, 'OOD2')

df = pd.concat([indomain_train, indomain_test, ood1, ood2])

df.to_csv('./saved_everything/'+ str(args.dataset) +'/dataset_stats.csv')


######################## 1. bert predictive resultes -- on In domain / ood1 / ood2

Full_data = pd.read_json('./models/' + str(args.dataset) + '_full/bert_predictive_performances.json')
Full_data['domain'] = 'Full size'
InDomain = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances.json')
InDomain['domain'] = 'InDomain'

path = os.path.join('./models/', str(args.dataset),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
OOD1 = pd.read_json(path)
OOD1['domain'] = 'OOD1'

path = os.path.join('./models/', str(args.dataset),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
OOD2 = pd.read_json(path)
OOD2['domain'] = 'OOD2'

result = pd.concat([Full_data, InDomain, OOD1, OOD2])
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
    df_tf.columns = df.columns  # ['random','scaled_attention','attention','ig','lime','gradients','deeplift']

    df_tf['Rationales_metrics'] = ['Sufficiency', 'Comprehensiveness', 'AOPC_sufficiency', 'AOPC_comprehensiveness']
    df_tf['Domain'] = str(domain)
    df_tf = df_tf.set_index('Rationales_metrics')
    return df_tf





seed_list = []
for seed in [5,10,15,20,25]:
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
        df1 = json2df(json, 'OOD1')
        OOD2 = pd.read_json(ood2_path)
        df2 = json2df(json, 'OOD2')

        final = pd.concat([df, df1, df2], ignore_index=False)
        final['thresholder'] = str(thresh)
        df_list.append(final)

    seed_n = pd.concat([df_list[0], df_list[1]], ignore_index=False)
    seed_n['seed'] = seed
    seed_list.append(seed_n)

posthoc_faithfulness = pd.concat([seed_list[0],seed_list[1],seed_list[2],seed_list[3],seed_list[4]], ignore_index=False)
posthoc_faithfulness.to_csv('saved_everything/' + str(args.dataset) + '/posthoc_faithfulness.csv')

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

