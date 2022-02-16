import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

df = pd.read_json("./models/WS/model_run_stats/test-pred_prob-bert:25.json").transpose()
df['annotation_id'] = df.index
df_attention = pd.read_json("./extracted_rationales/WS/data/contigious/attention-test.json", orient='records')
print(df_attention)

df_gradients = pd.read_json("./extracted_rationales/WS/data/contigious/gradients-test.json", orient='records')
df_attention=df_attention.rename(columns={"text": "attention_rationales"})
print(df_attention)

df_gradients=df_gradients.rename(columns={"text": "gradients_rationales"})


df_attention = df_attention.drop(['exp_split', 'label', 'label_id'], axis=1)
df_gradients = df_gradients.filter(['annotation_id', 'gradients_rationales'], axis=1)

print(df_attention)

new = pd.merge(df_gradients, df_attention, on = 'annotation_id')
new = pd.merge(new, df, on = 'annotation_id')
print(new)
new.to_csv('WS_results_with_rationales.csv')
