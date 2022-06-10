import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns




#method = 'Blob' 
method = 'Vader'
data = 'factcheck'
ALPHA = 0.4




test = pd.read_json('./datasets/'+str(data)+'/data/test.json')
ood1 = pd.read_json('./datasets/'+str(data)+'_ood1/data/test.json')
ood2 = pd.read_json('./datasets/'+str(data)+'_ood2/data/test.json')

def generate_sent_list(df, method):
    if method == 'Vader':
        sent = SentimentIntensityAnalyzer()
        polarity = [round(sent.polarity_scores(i)['compound'], 2) for i in df['text']]
    else:
        polarity = []
        for t in df['text']:
            blob = TextBlob(t)
            sent = blob.sentiment.polarity
            polarity.append(sent)
    df['sentiment_score'] = polarity
    return df



test = generate_sent_list(test, method)
ood1 = generate_sent_list(ood1, method)
ood2 = generate_sent_list(ood2, method)





sns.set(style="darkgrid")
sns.color_palette("hls", 8)

fig = sns.kdeplot(test['sentiment_score'], shade=False, color="r")
fig = sns.kdeplot(ood1['sentiment_score'], shade=True, color="b")
fig = sns.kdeplot(ood2['sentiment_score'], shade=True, color="y")
plt.show()

# plt.hist(test['sentiment_score'], label='test', alpha=ALPHA+0.4)
# plt.hist(ood1['sentiment_score'], label='ood1', alpha=ALPHA)
# plt.hist(ood2['sentiment_score'], label='ood2', alpha=ALPHA)
# plt.legend()



fig = plt.gcf()
fig.savefig('./sent_distribution/'+str(data)+str(method)+'.png', dpi=250)
# plt.show()