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
import numpy as np



parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "complain",
)

parser.add_argument(
    "--combine_all_tasks",
    help='combine all',
    action='store_true',
    default=False
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
    '--save_accuracy_version',
    help='decide which parts are in need',
    action='store_true',
    default=False
)




### 最完整的版本，直接提json的结果 最近期的
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)

if args.combine_all_tasks:
    task_list = ['agnews', 'xfact', 'factcheck', 'AmazDigiMu', 'AmazPantry', 'yelp']
    result_list = []
    for task in task_list:
        result = pd.read_csv('saved_everything/' + str(task) + '/selective_results.csv')
        result['Task'] = str(task)

        result_list.append(result)
    final = pd.concat(result_list)
    print(final)
    print(final.dtypes)
    final.to_csv('./saved_everything/test.csv')

    exit()





if args.save_accuracy_version:
    select_columns = ['mean-acc', 'std-acc']
else:
    select_columns = ['mean-f1', 'std-f1']




######################## 1. bert predictive resultes -- on In domain / ood1 / ood2
InDomain = pd.read_json('./models/'+str(args.dataset)+'/bert_predictive_performances.json')
Full_data = pd.read_json('./models/'+str(args.dataset)+'_full/bert_predictive_performances.json')

path = os.path.join('./models/', str(args.dataset),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
OOD1 = pd.read_json(path)
path = os.path.join('./models/', str(args.dataset),'bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
OOD2 = pd.read_json(path)


Full_data = Full_data[select_columns].iloc[0]
InDomain = InDomain[select_columns].iloc[0]
OOD1 = OOD1[select_columns].iloc[0]
OOD2 = OOD2[select_columns].iloc[0]


Full_data['Domain'] = 'Full'
InDomain['Domain'] = 'SynD'
OOD1['Domain'] = 'AsyD1'
OOD2['Domain'] = 'AsyD2'
bert_result = pd.concat([Full_data, InDomain, OOD1, OOD2], ignore_index=False, axis=1).T
cols = bert_result.columns.tolist()
cols = cols[-1:] + cols[:-1]
bert_result = bert_result[cols]
if args.save_accuracy_version:
    bert_result = bert_result.reset_index()[['Domain', 'mean-acc', 'std-acc']]
    bert_result = bert_result.rename(columns={"mean-acc":"BERT ACC", "std-acc":"BERT ACC"})
else:
    bert_result = bert_result.reset_index()[['Domain', 'mean-f1', 'std-f1']]
    bert_result = bert_result.rename(columns={"mean-f1":"BERT F1", "std-f1":"BERT std"})

bert_result['BERT F1'] = bert_result['BERT F1'].astype(float, errors = 'raise')
bert_result['BERT std'] = bert_result['BERT std'].astype(float, errors = 'raise')
#bert_result.to_csv('./saved_everything/'+str(args.dataset)+'/bert_predictive.csv')  # less column
####################################################################################
#####################################################################################



########################### 3. FRESH results of top scaled attention
import os.path


if args.dataset == 'AmazDigiMu':
    fresh_OOD1 = pd.read_json('FRESH_classifiers/AmazDigiMu/topk/scaled attention_bert_predictive_performances-OOD-AmazDigiMu_ood1.json')
    fresh_OOD2 = pd.read_json('FRESH_classifiers/AmazDigiMu/topk/scaled attention_bert_predictive_performances-OOD-AmazDigiMu_ood2.json')
    fresh_full_data = pd.read_json('./FRESH_classifiers/AmazDigiMu_full/topk/scaled attention_bert_predictive_performances.json')
    fresh_InDomain = pd.read_json('./FRESH_classifiers/AmazDigiMu/topk/scaled attention_bert_predictive_performances.json')

elif args.dataset == 'AmazPantry':
    fresh_OOD1 = pd.read_json('FRESH_classifiers/AmazPantry/topk/scaled attention_bert_predictive_performances-OOD-AmazPantry_ood1.json')
    fresh_OOD2 = pd.read_json('FRESH_classifiers/AmazPantry/topk/scaled attention_bert_predictive_performances-OOD-AmazPantry_ood2.json')
    fresh_full_data = pd.read_json('./FRESH_classifiers/AmazPantry_full/topk/scaled attention_bert_predictive_performances.json')
    fresh_InDomain = pd.read_json('./FRESH_classifiers/AmazPantry/topk/scaled attention_bert_predictive_performances.json')

else:
    fresh_OOD1_path = os.path.join('FRESH_classifiers', str(args.dataset), 'topk/', 'scaled attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
    fresh_OOD2_path = os.path.join('FRESH_classifiers', str(args.dataset), 'topk/', 'scaled attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
    try:
        file_exists = os.path.exists(fresh_OOD1_path)
    except:
        fresh_OOD1_path = os.path.join('FRESH_classifiers', str(args.dataset), 'topk/', 'scaled_attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
        fresh_OOD2_path = os.path.join('FRESH_classifiers', str(args.dataset), 'topk/', 'scaled_attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
        file_exists = os.path.exists(fresh_OOD1_path)
    print('fresh ood path ')
    print(fresh_OOD1_path)
    fresh_OOD1 = pd.read_json(fresh_OOD1_path)
    fresh_OOD2 = pd.read_json(fresh_OOD2_path)




    fresh_full_data = pd.read_json(
        './FRESH_classifiers/'+str(args.dataset)+'_full/topk/scaled attention_bert_predictive_performances.json')
    fresh_InDomain = pd.read_json(
        './FRESH_classifiers/' + str(args.dataset) + '/topk/scaled attention_bert_predictive_performances.json')


fresh_full_data = fresh_full_data[select_columns].iloc[1]
fresh_full_data['Domain'] = 'Full size'


fresh_InDomain = fresh_InDomain[select_columns].iloc[1]
fresh_InDomain['Domain'] = 'SynD'


fresh_OOD1 = fresh_OOD1[select_columns].iloc[1]
fresh_OOD1['Domain'] = 'AsyD1'


fresh_OOD2 = fresh_OOD2[select_columns].iloc[1]
fresh_OOD2['Domain'] = 'AsyD2'


fresh_result = pd.concat([fresh_full_data, fresh_InDomain, fresh_OOD1, fresh_OOD2], axis=1, ignore_index=False).T.reset_index()[select_columns]
if args.save_accuracy_version:
    fresh_result = fresh_result.rename(columns={"mean-acc":"FRESH ACC", "std-acc":"FRESH std"})
else:
    fresh_result = fresh_result.rename(columns={"mean-f1":"FRESH F1", "std-f1":"FRESH std"})

fresh_result['FRESH F1'] = fresh_result['FRESH F1'].astype(float, errors = 'raise')
fresh_result['FRESH std'] = fresh_result['FRESH std'].astype(float, errors = 'raise')
#fresh_result.to_csv('./saved_everything/'+str(args.dataset)+'/bert_predictive.csv')



####################################  KUMA AND LSTM ############################################################

## get KUMA of FULL / IN D / OOD1 / OOD2
kuma_FullData = pd.read_json('./kuma_model/'+str(args.dataset)+'_full/kuma-bert_predictive_performances.json')
# pd.options.display.max_columns = None
# print('    KUMA    FULL ')
# print(kuma_FullData)
# kuma_InDomain_path = os.path.join('kuma_model',str(args.dataset),'/kuma-bert_predictive_performances.json')
# print('    KUMA    kuma_InDomain_path ')
# print(kuma_InDomain_path)
kuma_InDomain = pd.read_json('./kuma_model/'+str(args.dataset)+'/kuma-bert_predictive_performances.json')
kuma_OOD1 = pd.read_json('./kuma_model/'+str(args.dataset)+'/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
kuma_OOD2 = pd.read_json('./kuma_model/'+str(args.dataset)+'/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
print('    KUMA    OOD1 ')
print(kuma_OOD1)

LSTM_FullData = pd.read_json(
    './LSTM_model/' + str(args.dataset) + '_full/full_lstm-bert_predictive_performances.json')
LSTM_InDomain = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances.json')
LSTM_OOD1 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(
    args.dataset) + '_ood1.json')
LSTM_OOD2 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(
    args.dataset) + '_ood2.json')


kuma_FullData = kuma_FullData[select_columns].iloc[0]
kuma_InDomain = kuma_InDomain[select_columns].iloc[0]
kuma_OOD1 = kuma_OOD1[select_columns].iloc[0]
kuma_OOD2 = kuma_OOD2[select_columns].iloc[0]

LSTM_FullData = LSTM_FullData[select_columns].iloc[0]
LSTM_InDomain = LSTM_InDomain[select_columns].iloc[0]
LSTM_OOD1 = LSTM_OOD1[select_columns].iloc[0]
LSTM_OOD2 = LSTM_OOD2[select_columns].iloc[0]


kuma_result = pd.concat([kuma_FullData, kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False, axis=1).T
kuma_result = kuma_result.reset_index()[select_columns]
LSTM_result = pd.concat([LSTM_FullData, LSTM_InDomain, LSTM_OOD1, LSTM_OOD2], ignore_index=False, axis=1).T
LSTM_result = LSTM_result.reset_index()[select_columns]

if args.save_accuracy_version:
    kuma_result = kuma_result.rename(columns={"mean-acc":"KUMA ACC", "std-acc":"KUMA std"})
    LSTM_result = LSTM_result.rename(columns={"mean-acc":"LSTM ACC", "std-acc":"LSTM std"})
else:
    kuma_result = kuma_result.rename(columns={"mean-f1":"KUMA F1", "std-f1":"KUMA std"})
    LSTM_result = LSTM_result.rename(columns={"mean-f1":"LSTM F1", "std-f1":"LSTM std"})

def one_domain_len(domain):
    overall = np.zeros(5)
    for _j_, seed in enumerate([5,10,15,20,25]):
        if 'ood' in str(domain):
            path_to_file : str = f'kuma_model/{args.dataset}/kuma-bert-output_seed-kuma-bert{seed}-OOD-{args.dataset}_{domain}.npy'
        elif 'full' in str(domain):
            path_to_file : str = f'kuma_model/{args.dataset}_full/kuma-bert-output_seed-kuma-bert{seed}.npy'
        else:
            path_to_file : str = f'kuma_model/{args.dataset}/kuma-bert-output_seed-kuma-bert{seed}.npy'
        
        file_data = np.load(path_to_file, allow_pickle=True).item()
        aggregated_ratio = np.zeros(len(file_data))

        for _i_, (docid, metadata) in enumerate(file_data.items()):

            rationale_ratio = min( 
                1.,
                metadata['rationale'].sum()/metadata['full text length']
            )

            aggregated_ratio[_i_] = rationale_ratio

        overall[_j_] = aggregated_ratio.mean()
    
    return overall.mean(), overall.std(), overall


Full_len_mean, Full_len_std, Full_len_overall = one_domain_len('full')
InDomain_len_mean, InDomain_len_std, InDomain_len_overall = one_domain_len('InDomain')
ood1_len_mean, ood1_len_std, ood1_len_overall= one_domain_len('ood1')
ood2_len_mean, ood2_len_std, ood2_len_overall= one_domain_len('ood2')

kuma_result['KUMA Len'] = [Full_len_mean,InDomain_len_mean,ood1_len_mean,ood2_len_mean]

#############################
SPECTRA = pd.read_csv('saved_everything/' + str(args.dataset) + '/spectra_mean.csv')[['avg', 'std']].rename(columns={"avg":"SPECTRA F1", "std":"SPECTRA std"})
##############################

final = pd.concat([bert_result, fresh_result, LSTM_result, kuma_result, SPECTRA], axis=1)
final['Domain'] = ['Full', 'SynD', 'AsyD1', 'AsyD2']
final = final.rename({'Domain': 'Testing Set'})
s = final[final.select_dtypes(exclude=['object']).columns] * 100
final[s.columns] = s
print(final.dtypes)
print(final)
if args.save_accuracy_version:
    final.to_csv('saved_everything/' + str(args.dataset) + '/selective_results_acc.csv')
else:    
    final.to_csv('saved_everything/' + str(args.dataset) + '/selective_results.csv')

