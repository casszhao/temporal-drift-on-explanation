import pandas as pd
import os
import argparse




parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "complain",
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



## get KUMA of FULL / IN D / OOD1 / OOD2
kuma_FullData = pd.read_json('./kuma_model/' + str(args.dataset) + '_full/kuma-bert_predictive_performances.json')
kuma_InDomain = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances.json')
kuma_OOD1 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
kuma_OOD2 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')

LSTM_FullData = pd.read_json('./LSTM_model/' + str(args.dataset) + '_full/full_lstm-bert_predictive_performances.json')
LSTM_InDomain = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances.json')
LSTM_OOD1 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
LSTM_OOD2 = pd.read_json('./LSTM_model/' + str(args.dataset) + '/full_lstm-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')

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


kuma_FullData['domain'] = 'Full data'
kuma_InDomain['domain'] = 'InDomain'
kuma_OOD1['domain'] = 'OOD1'
kuma_OOD2['domain'] = 'OOD2'

LSTM_FullData['domain'] = 'Full data'
LSTM_InDomain['domain'] = 'InDomain'
LSTM_OOD1['domain'] = 'OOD1'
LSTM_OOD2['domain'] = 'OOD2'

kuma_result = pd.concat([kuma_FullData, kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False, axis = 1).T
kuma_result['Model'] = 'Kuma'
print(kuma_result)

LSTM_result = pd.concat([LSTM_FullData, LSTM_InDomain, LSTM_OOD1, LSTM_OOD2], ignore_index=False, axis = 1).T
LSTM_result['Model'] = 'LSTM'

final = pd.concat([kuma_result, LSTM_result], ignore_index=False)
final.to_csv('saved_everything/' + str(args.dataset) + '/KUMA_LSTM_predictive_results.csv')
