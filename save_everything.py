# 1. bert predictive resultes -- on In domain / ood1 / ood2
# 2. different measures of different attributes rationales for both top / contigious -- on In domain / ood1 / ood2
# 3. FRESH results
# 4. kuma results
# 5. domain similarity between:  In domain / ood1 / ood2
# 6. rationale similarity between:  In domain / ood1 / ood2
# 7. datasets metadata: train/test/ size, time span, label distribution

import pandas as pd
import json
import csv
import config.cfg
import os
import argparse




parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "complain",
)
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)





######################## 1. bert predictive resultes -- on In domain / ood1 / ood2

Full_data = pd.read_json('./models/' + str(args.dataset) + '_full/bert_predictive_performances.json')
Full_data['domain'] = 'Full size'
InDomain = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances.json')
InDomain['domain'] = 'InDomain'
OOD1 = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances-OOD-complain_ood1.json')
OOD1['domain'] = 'OOD1'
OOD2 = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances-OOD-complain_ood2.json')
OOD2['domain'] = 'OOD2'
result = pd.concat([Full_data, InDomain, OOD1, OOD2])
result.to_csv('saved_everything/' + str(args.dataset) + '/bert_predictive_on_fulltext.csv')

####################################################################################


######################################## 2. faithful of different measures & different attributes rationales for both top / contigious -- on In domain / ood1 / ood2 #########################

def json2df(df, domain):
    df.rename(columns={"": "Task"})
    list_of_list = []
    for col in range(0, 7):
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
        json = pd.read_json('./posthoc_results/' + str(args.dataset) + '/' + str(thresh) + '-faithfulness-scores-averages-5-description.json')
        df = json2df(json, 'InDomain')

        json1 = pd.read_json('./posthoc_results/' + str(args.dataset) + '/' + str(thresh) + '-faithfulness-scores-averages-OOD-complain_ood1-5-description.json')
        df1 = json2df(json, 'OOD1')

        json2 = pd.read_json('./posthoc_results/' + str(args.dataset) + '/' + str(thresh) + '-faithfulness-scores-averages-OOD-complain_ood2-5-description.json')
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
    for attribute_name in ["attention", "ig", "gradients", "lime", "deeplift", "scaled_attention"]:
        path = os.path.join('FRESH_classifiers/', str(args.dataset), str(threshold),
                            str(attribute_name) + '_bert_predictive_performances.json')
        #fresh_InDomain = pd.read_json('FRESH_classifiers/complain/topk/attention_bert_predictive_performances.json')
        fresh_InDomain = pd.read_json(path)

        fresh_InDomain = fresh_InDomain[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
        fresh_InDomain['domain'] = 'InDomain'

        path1 = os.path.join('FRESH_classifiers/', str(args.dataset), str(threshold), str(attribute_name) + '_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
        # fresh_OOD1 = pd.read_json(path1)
        path1 = './FRESH_classifiers/complain/topk/attention_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json'
        fresh_OOD1 = pd.read_json(path1)

        # fresh_OOD1 = pd.read_json('./FRESH_classifiers/' + str(args.dataset) + '/topk/' + str(attribute_name) + '_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
        fresh_OOD1 = fresh_OOD1[['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece']].iloc[1]
        fresh_OOD1['domain'] = 'OOD1'

        fresh_OOD2 = pd.read_json(os.path.join('./FRESH_classifiers/', str(args.dataset), str(threshold),
                                               str(attribute_name) + '_bert_predictive_performances-OOD-' + str(
                                                   args.dataset) + '_ood2.json'))

        # fresh_OOD2 = pd.read_json('./FRESH_classifiers/' + str(args.dataset) + '/' + str(threshold) + '/' + str(attribute_name) + '_bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
        fresh_OOD2 = fresh_OOD2[select_columns].iloc[1]
        fresh_OOD2['domain'] = 'OOD2'

        attribute_df = pd.concat([fresh_InDomain, fresh_OOD1, fresh_OOD2], axis=1, ignore_index=False).T.reset_index()[
            ['mean-acc', 'std-acc', 'mean-f1', 'std-f1', 'mean-ece', 'std-ece', 'domain']]
        attribute_df['attribute_name'] = str(attribute_name)
        attribute_list.append(attribute_df)

    attribute_results = pd.concat([attribute_list[0], attribute_list[1], attribute_list[2], attribute_list[3], attribute_list[4], attribute_list[5]], ignore_index=False)
    attribute_results['threshold'] = str(threshold)
    thresh_hold_list.append(attribute_results)

fresh_final_result = pd.concat([thresh_hold_list[0], thresh_hold_list[1]], ignore_index=False)
fresh_final_result.to_csv('saved_everything/' + str(args.dataset) + '/fresh_predictive_results.csv')



############################################################ 4. kuma predictive results


kuma_InDomain = pd.read_json('./kuma_model_new/' + str(args.dataset) + '/kuma-bert_predictive_performances.json')
kuma_InDomain['domain'] = 'InDomain'
kuma_OOD1 = pd.read_json('./kuma_model_new/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood1.json')
kuma_OOD1['domain'] = 'OOD1'
kuma_OOD2 = pd.read_json('./kuma_model_new/' + str(args.dataset) + '/kuma-bert_predictive_performances-OOD-' + str(args.dataset) + '_ood2.json')
kuma_OOD2['domain'] = 'OOD2'
kuma_result = pd.concat([kuma_InDomain, kuma_OOD1, kuma_OOD2], ignore_index=False)
kuma_result.to_csv('saved_everything/' + str(args.dataset) + '/kuma_predictive_on_fulltext.csv')

################################################################################################

