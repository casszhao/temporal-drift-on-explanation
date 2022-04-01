import pandas as pd
import os
import argparse




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




kuma_InDomain = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances.json')
kuma_InDomain['domain'] = 'InDomain'
kuma_OOD1 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
kuma_OOD1['domain'] = 'OOD1'
kuma_OOD2 = pd.read_json('./kuma_model/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
kuma_OOD2['domain'] = 'OOD2'
kuma_result = pd.concat([kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False)
kuma_result.to_csv('saved_everything/' + str(args.dataset) + '/kuma_predictive_on_fulltext.csv')