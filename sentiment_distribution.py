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
import scipy


def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    avg_repr = 0.5 * (repr1 + repr2) # M
    sim = 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

task_list = ['agnews','xfact','factcheck','AmazDigiMu','AmazPantry','yelp']

#method = 'Blob' 
method = 'Vader'
data = 'xfact'
ALPHA = 0.4

def KL(P,Q):

     epsilon = 0.00001
     # You may want to instead make copies to avoid changing the np arrays.
     P = P+epsilon
     Q = Q+epsilon
     divergence = np.sum(P*np.log(P/Q))
     return divergence



# example of calculating the kl divergence (relative entropy) with scipy
from scipy.special import rel_entr
# define distributions
# calculate (P || Q)

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



def get_sent_diver(data):

#train = pd.read_json('./datasets/'+str(data)+'/data/train.json')
    test = pd.read_json('./datasets/'+str(data)+'/data/test.json')
    ood1 = pd.read_json('./datasets/'+str(data)+'_ood1/data/test.json')
    ood2 = pd.read_json('./datasets/'+str(data)+'_ood2/data/test.json')

    #train = generate_sent_list(train, method)
    test = generate_sent_list(test, method)
    ood1 = generate_sent_list(ood1, method)
    ood2 = generate_sent_list(ood2, method)

    print(ood2)

    # sns.set(style="darkgrid")
    # sns.color_palette("hls", 8)

    #fig = sns.kdeplot(train['sentiment_score'], shade=False, color="r")
    # fig = sns.kdeplot(test['sentiment_score'], shade=False, label = 'SynD', color="g")
    # fig = sns.kdeplot(ood1['sentiment_score'], shade=True, label = 'AsyD1', color="b")
    # fig = sns.kdeplot(ood2['sentiment_score'], shade=True, label = 'AsyD2', color="y")
    # plt.show()
    # fig = plt.gcf()
    # fig.savefig('./sent_distribution/'+str(data)+str(method)+'.png', dpi=350)
    # plt.legend()

    density_t, bins_t, patches_t = plt.hist(test['sentiment_score'], label='test', alpha=ALPHA+0.4)
    density1, bins1, patches1 = plt.hist(ood1['sentiment_score'], label='ood1', alpha=ALPHA)
    density2, bins2, patches2 = plt.hist(ood2['sentiment_score'], label='ood2', alpha=ALPHA)
    #plt.legend()

    print(density1)
    sent_diverge1 = jensen_shannon_divergence(density1, density_t)
    sent_diverge2 = jensen_shannon_divergence(density2, density_t)
    print(sent_diverge1, sent_diverge2)

    return sent_diverge1, sent_diverge2

sent_diver_list = []
for task in task_list:
    sent_diverge1, sent_diverge2 = get_sent_diver(task)
    sent_diver_list.append(sent_diverge1)
    sent_diver_list.append(sent_diverge2)

all_task_all_sentiment_and_corre = pd.read_csv('./saved_everything/all_tasks_all_factors_onlyAsyD.csv')
all_task_all_sentiment_and_corre['Sent Div'] = sent_diver_list
all_task_all_sentiment_and_corre = all_task_all_sentiment_and_corre.rename(columns={
    'Suff changes':'Suff Diff','Comp changes':'Comp Diff','Temporal distance': 'Temp Diff','Corpus divergence':'Corp Diff','Text length':'Text Len'})

all_task_all_sentiment_and_corre.to_csv('./saved_everything/all_task_all_sentiment_and_corre.csv')

# plt.show()