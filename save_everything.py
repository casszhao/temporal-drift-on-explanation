# 1. bert predictive resultes -- on In domain / ood1 / ood2
# 2. different measures of different attributes rationales for both top / contigious -- on In domain / ood1 / ood2
# 3. FRESH results
# 4. kuma results
# 5. domain similarity between:  In domain / ood1 / ood2
# 6. rationale similarity between:  In domain / ood1 / ood2
# 7. datasets metadata: train/test/ size, time span, label distribution
import argparse
import os
import gc
import pandas as pd
import config.cfg




parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "complain",
)
args = parser.parse_args()
# 1. bert predictive resultes -- on In domain / ood1 / ood2

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)

# df = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances.json')
InDomain = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances.json')
InDomain['domain'] = 'InDomain'
OOD1 = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances-OOD-complain_ood1.json')
OOD1['domain'] = 'OOD1'
print(OOD1)
OOD2 = pd.read_json('./models/' + str(args.dataset) + '/bert_predictive_performances-OOD-complain_ood2.json')
OOD2['domain'] = 'OOD2'
result = pd.concat([InDomain, OOD1, OOD2])
result.to_csv('saved_everything/' + str(args.dataset) + '/bert_predictive_on_fulltext.csv')



# 2. different measures of different attributes rationales for both top / contigious -- on In domain / ood1 / ood2




# 3. FRESH results
# 4. kuma results
# 5. domain similarity between:  In domain / ood1 / ood2
# 6. rationale similarity between:  In domain / ood1 / ood2
# 7. datasets metadata: train/test/ size, time span, label distribution