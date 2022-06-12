# 5. domain similarity between:  In domain / ood1 / ood2
## 6. rationale similarity between:  In domain / ood1 / ood2
import pandas as pd
import os
import argparse
import logging
import pickle
import copy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import robo
#from robo.fmin import bayesian_optimization

import task_utils
import data_utils
import similarity
import features
from constants import FEATURE_SETS, SENTIMENT, POS, POS_BILSTM, PARSING,\
    TASK2TRAIN_EXAMPLES, TASK2DOMAINS, TASKS, POS_PARSING_TRG_DOMAINS,\
    SENTIMENT_TRG_DOMAINS, BASELINES, BAYES_OPT, RANDOM, MOST_SIMILAR_DOMAIN,\
    MOST_SIMILAR_EXAMPLES, ALL_SOURCE_DATA, SIMILARITY_FUNCTIONS


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type = str,
    help = "select dataset / task",
    default = "factcheck",
)

parser.add_argument(
    '--combine_all',
    help='combine all dataset',
    action='store_true',
    default= False
)

parser.add_argument(
    '--use_saved_simi',
    help='combine all dataset',
    action='store_true',
    default= False
)
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset) + '/'
os.makedirs(datasets_dir, exist_ok = True)

similarity_method = 'Topic jensen-shannon' # jensen-shannon renyi cosine euclidean variational bhattacharyya
# term js kind works
# 'Topic bhattacharyya' works
# Term bhattacharyya' similar to Topic bhattacharyya'

print(' ======================= ')

if args.combine_all:
    task_list = ['agnews','xfact','factcheck','AmazDigiMu','AmazPantry','yelp']
    df_list = []
    full_simi_df_list = []
    for task in task_list:
        df = pd.read_csv('./saved_everything/'+str(task)+'/corre_table_'+str(similarity_method)+'.csv')[['AsyD1', 'AsyD2']].T
        df.columns = ['suff_diff','comp_diff','temporal_distance','corpus_similarity', 'text avg length']
        df['Task'] = str(task)

        for fname in os.listdir('./saved_everything/'+str(task)+'/'):
            if 'fulltext_similarity_vocab' in fname:
                
                full_simi = pd.read_csv('./saved_everything/'+str(task)+'/'+fname)
                full_simi['Task'] = str(task)
                #full_simi['Domain'] = ['SynD', 'AsyD1', 'AsyD2']
                print(full_simi)
        df_list.append(df)
        full_simi_df_list.append(full_simi)

    df = pd.concat(df_list)
    full_simi=pd.concat(full_simi_df_list)
    full_simi.to_csv('./saved_everything/all_tasks_full_similarity.csv')
    
    df.to_csv('./saved_everything/all_tasks_all_factors_onlyAysD.csv')
    print('+++++++++++++')
    print(df)
<<<<<<< HEAD
    corr = df.corr(method='spearman')
    #corr.to_csv('/saved_everything/'+str(similarity_method)+'_all.csv')
    corr.style.background_gradient(cmap='coolwarm')
=======
    df.columns = ['suff_diff','comp_diff','temporal_distance','corpus_similarity']
    corr = df.corr()
>>>>>>> 8da041495d84804af3318e14f16934b6e696ab9c
    print(corr)
    plt.figure(figsize=(16, 6))
    import seaborn as sns
    import matplotlib.pyplot as plt
# Store heatmap object in a variable to easily access it when you want to include more features (such as title).
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)

    exit()


def task2_objective_function(task):
    """Returns the objective function of a task."""
    if task == SENTIMENT:
        return objective_function_sentiment
    if task == POS:
        return objective_function_pos
    if task == POS_BILSTM:
        return objective_function_pos_bilstm
    if task == PARSING:
        return objective_function_parsing
    raise ValueError('No objective function implemented for %s.' % task)


def objective_function_sentiment(feature_weights):
    """
    The objective function to optimize for sentiment analysis.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, SENTIMENT,
        TASK2TRAIN_EXAMPLES[SENTIMENT])

    # train and evaluate the SVM; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_sentiment(
        train_subset, train_labels_subset, X_val, y_val, X_test, y_test)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_pos(feature_weights):
    """
    The objective function to optimize for POS tagging.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, POS,
        TASK2TRAIN_EXAMPLES[POS])

    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_pos(
        train_subset, train_labels_subset, X_val, y_val)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_pos_bilstm(feature_weights):
    """
    The objective function to optimize for POS tagging.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, POS_BILSTM,
        TASK2TRAIN_EXAMPLES[POS_BILSTM])

    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_pos_bilstm(
        train_subset, train_labels_subset, X_val, y_val)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_parsing(feature_weights):
    """
    The objective function to optimize for dependency parsing.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, PARSING,
        TASK2TRAIN_EXAMPLES[PARSING])
    val_accuracy, _ = task_utils.train_and_evaluate_parsing(
        train_subset, train_labels_subset, X_val, y_val,
        parser_output_path=parser_output_path,
        perl_script_path=perl_script_path)
    error = 100 - float(val_accuracy)
    return error



def convert_to_listoflisttoken(text_full_list):
    list_of_list_of_tokens = []
    for sent in text_full_list:
        if str(sent) != 'nan':
            # print('-------sent: ', sent)
            try:
                list_of_tokens = sent.split()
                # print('==========list_of_tokens: ', list_of_tokens)
                list_of_list_of_tokens.append(list_of_tokens)
            except:
                print('-------sent cannot be split: ', sent)
    return list_of_list_of_tokens



def get_similarity_between_2reps(domain1, domain2, feature_names):

    domain1_term_dist, domain1_topic_dist = domain1
    domain2_term_dist, domain2_topic_dist = domain2
    # features here are actually similarities value betweeen two distributions
    Representations = []
    # Measures = []
    Similarity = []

    for j, f_name in enumerate(feature_names):
        # check whether feature belongs to similarity-based features,
        # diversity-based features, etc.
        # print(j)
        # print(f_name)
        # Measures.append(f_name)

        if f_name.startswith('topic'):
            f = similarity.similarity_name2value(
                f_name.split('_')[1], domain1_topic_dist, domain2_topic_dist)
            Representations.append('Topic distribution')

        # elif f_name.startswith('word_embedding'):
        #     f = similarity.similarity_name2value(
        #         f_name.split('_')[2], word_reprs[i], trg_word_repr)
        elif f_name in SIMILARITY_FUNCTIONS:
            f = similarity.similarity_name2value(
                f_name, domain1_term_dist, domain2_term_dist)
            Representations.append('Term distribution')
        # elif f_name in DIVERSITY_FEATURES:
        #     f = diversity_feature_name2value(
        #         f_name, examples[i], train_term_dist, vocab.word2id, word2vec)
        else:
            raise ValueError('%s is not a valid feature name.' % f_name)
        assert not np.isnan(f), 'Error: Feature %s is nan.' % f_name
        assert not np.isinf(f), 'Error: Feature %s is inf or -inf.' % f_name

        Similarity.append(f)

    return pd.DataFrame(list(zip(Similarity, Representations)),
                        columns=['Similarity', 'Representations'])




def pre_post_process(InD_test_reps, domain_reps, domain_column_name):
    df = get_similarity_between_2reps(InD_test_reps, domain_reps, feature_names)
    df['Measure'] = Measure
    df['Rep_Mea'] = Rep_Mea
    df['Domain'] = str(domain_column_name)
    return df



Rep_Mea = ['Term jensen-shannon', 'Term renyi', 'Term cosine', 'Term euclidean', 'Term variational', 'Term bhattacharyya',
           'Topic jensen-shannon', 'Topic renyi', 'Topic cosine', 'Topic euclidean', 'Topic variational', 'Topic bhattacharyya']

Measure = ['jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya',
           'jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']


feature_set_names = ['similarity', 'topic_similarity']
feature_names = features.get_feature_names(feature_set_names)
# for topic modelling:



##### get suff and comp difference
suff_diff = pd.read_csv('./saved_everything/'+str(args.dataset)+'/posthoc_faithfulness_overleaf.csv')
suff_In = suff_diff.iloc[1,1]  # row, column
suff_ood1 = suff_diff.iloc[2,1]
suff_ood2 = suff_diff.iloc[3,1]

comp_In = suff_diff.iloc[1,9]
comp_ood1 = suff_diff.iloc[2,9]
comp_ood2 = suff_diff.iloc[3,9]

suff_diff_1 = abs(suff_In - suff_ood1)
suff_diff_2 = abs(suff_In - suff_ood2)
comp_diff_1 = abs(comp_In - comp_ood1)
comp_diff_2 = abs(comp_In - comp_ood2)

faith_scores = pd.DataFrame({'AsyD1': [suff_diff_1, comp_diff_1], 'AsyD2': [suff_diff_2, comp_diff_2]})
print(' =========== ' + str(args.dataset) + '===========')
index_faithful = ['Suff_diff', 'Comp_diff']

# corre_table.to_csv('./saved_everything/' + str(args.dataset) + '/faith_diff.csv')
# exit()



indomain = pd.read_json('./datasets/'+str(args.dataset)+'/data/test.json')
ood1 = pd.read_json('./datasets/'+str(args.dataset)+'_ood1/data/test.json')
ood2 = pd.read_json('./datasets/'+str(args.dataset)+'_ood2/data/test.json')


############# text length

text_length1 = ood1['text'].apply(len).mean()
text_length2 = ood2['text'].apply(len).mean()
text_length = pd.DataFrame({'AsyD1': [text_length1], 'AsyD2': [text_length2]})
print( ' ---------- TEXT LEN -------- ' )
print(text_length)


############# get time different
def sort_dates(df):
    if "xfact" in str(args.dataset):
        df = df[pd.to_datetime(df['claimDate'], errors='coerce').notna()] # claimDate  for xfact
        df = df.dropna().sort_values(by='claimDate', na_position='first') # claimDate  for xfact
        df['date'] = pd.to_datetime(df['claimDate']).dt.date              # claimDate  for xfact
    else:            
        df['date'] = pd.to_datetime(df['date'], errors = 'coerce', utc=True).dt.date
        df = df.dropna().sort_values(by='date', na_position='first')
    return df

indomain = sort_dates(indomain) 
ood1 = sort_dates(ood1) 
ood2 = sort_dates(ood2) 
        
# label_dist = df['label'].value_counts().to_string()
# label_num = df['label'].nunique()


def get_time_span_info(df):
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]

    print(df['date'])
    print('---', start_date)
    print('---', end_date)
    quartile = int(len(df) * 0.25)

    DATE = df['date'].tolist()
    Interquartile_start = DATE[quartile]
    Interquartile_Mid = DATE[int(quartile*2)]
    Interquartile_end = DATE[-quartile]

    duration = end_date - start_date
    inter_duration = Interquartile_end - Interquartile_start
    print('---duration ---')
    print(duration)
    if int(duration.days) <= 0:
        start_date = df['date'][len(df) - 1]
        end_date = df['date'][0]
        duration = end_date - start_date
    if int(inter_duration.days) <= 0:
        Interquartile_start = DATE[-quartile]
        Interquartile_end = DATE[quartile]
        inter_duration = Interquartile_end - Interquartile_start
    return start_date, end_date, duration, Interquartile_start, Interquartile_end, inter_duration, Interquartile_Mid

indomain_start_date, indomain_end_date, indomain_duration, indomain_Interquartile_start, indomain_Interquartile_end, indomain_inter_duration, indomain_Mid_day = get_time_span_info(indomain)
ood1_start_date, ood1_end_date, ood1_duration, ood1_Interquartile_start, ood1_Interquartile_end, ood1_inter_duration, ood1_Mid_day = get_time_span_info(ood1)
ood2_start_date, ood2_end_date, ood2_duration, ood2_Interquartile_start, ood2_Interquartile_end, ood2_inter_duration, ood2_Mid_day = get_time_span_info(ood2)


def time_dist(start1, start2, end1, end2, mid1, mid2):
    diff = abs(start1-start2) + abs(end1-end2) #+ abs(mid1-mid2)
    return int(diff.days)

def time_density(duration1, durantion2):
    return abs(duration1-durantion2)

if args.dataset == 'agnews':
    ood1_temporal_dist = time_dist(ood1_start_date, indomain_start_date, ood1_end_date, indomain_end_date, indomain_Mid_day, ood1_Mid_day)
    ood2_temporal_dist = time_dist(ood2_start_date, indomain_start_date, ood2_end_date, indomain_end_date, indomain_Mid_day, ood2_Mid_day)
elif args.dataset == 'yelp':
    ood1_temporal_dist = time_dist(ood1_start_date, indomain_start_date, ood1_end_date, indomain_end_date, indomain_Mid_day, ood1_Mid_day)
    ood2_temporal_dist = time_dist(ood2_start_date, indomain_start_date, ood2_end_date, indomain_end_date, indomain_Mid_day, ood2_Mid_day)
else:
    ood1_temporal_dist = time_dist(ood1_Interquartile_start, indomain_Interquartile_start, ood1_Interquartile_end, indomain_Interquartile_end, indomain_Mid_day, ood1_Mid_day)
    ood2_temporal_dist = time_dist(ood2_Interquartile_start, indomain_Interquartile_start, ood2_Interquartile_end, indomain_Interquartile_end, indomain_Mid_day, ood2_Mid_day)

temporal_distance = pd.DataFrame(data={'AsyD1':[ood1_temporal_dist], 
                                       'AsyD2':[ood2_temporal_dist]})
                                       


index_time = ['temporal_distance']
# corre_table = pd.concat([corre_table,temporal_distance])



############################# domain similarity between:  In domain / ood1 / ood2
#########################################################################################
if args.use_saved_simi:
    pass
else:
    model_dir = './similarity_models/'+str(args.dataset)+'/'
    num_iterations = 2000 # 2000# for testing, original use 2000? need to check the paper
    VOCAB_SIZE = 20000    # 20000  
    os.makedirs(model_dir, exist_ok=True)

    InD_test_list = indomain['text']
    OOD1_list= ood1['text']
    OOD2_list= ood2['text']

    in_domain_test_list_list = convert_to_listoflisttoken(InD_test_list)
    OOD1_test_list_list = convert_to_listoflisttoken(OOD1_list)
    OOD2_test_list_list = convert_to_listoflisttoken(OOD2_list)

    list_of_list_of_tokens =  in_domain_test_list_list + OOD1_test_list_list + OOD2_test_list_list

    # create the vocabulary or load it if it was already created
    vocab_path = os.path.join(model_dir, 'vocab.txt')
    vocab = data_utils.Vocab(VOCAB_SIZE, vocab_path) # two functions, load and create
    vocab.create(list_of_list_of_tokens, lowercase=True)

    term_dist_path = os.path.join(datasets_dir, 'term_dist.txt')
    topic_vectorizer, lda_model = similarity.train_topic_model(in_domain_test_list_list, vocab, num_topics=50, num_iterations=num_iterations, num_passes=10)

    #InD_train_reps = features.get_reps_for_one_domain(in_domain_train_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True) # 0. term dist 1. topic dist
    InD_test_reps = features.get_reps_for_one_domain(in_domain_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)
    OOD1_reps = features.get_reps_for_one_domain(OOD1_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)
    OOD2_reps = features.get_reps_for_one_domain(OOD2_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)

    #baseline_similarity = pre_post_process(InD_test_reps, 'In Domain(Baseline)')
    OOD1_similarity = pre_post_process(InD_test_reps, OOD1_reps, 'OOD1')
    OOD2_similarity = pre_post_process(InD_test_reps, OOD2_reps, 'OOD2')
    results = pd.concat([OOD1_similarity,OOD2_similarity],ignore_index=True)


#results.to_csv(datasets_dir + '/fulltext_ood2indomain_similarity_vocab' + str(vocab.size) + ' .csv')
# use saved similarity directly
for fname in os.listdir(datasets_dir):
    if 'fulltext_ood2indomain_similarity_vocab' in fname:
        similarity_path = os.path.join(datasets_dir, fname)
        similairity_df = pd.read_csv(similarity_path)
ood1_term = similairity_df.loc[(similairity_df['Rep_Mea'] == str(similarity_method)) & (similairity_df['Domain'] == 'OOD1')]['Similarity'].item()
print(ood1_term)
ood2_term = similairity_df.loc[(similairity_df['Rep_Mea'] == str(similarity_method)) & (similairity_df['Domain'] == 'OOD2')]['Similarity'].item()
corpus_simi = pd.DataFrame({'AsyD1': [ood1_term], 'AsyD2': [ood2_term]})
print(corpus_simi)

















index_corpus_simi = ['corpus_similarity']
# for i in [faith_scores, temporal_distance, corpus_simi, text_length]:
#     print(i)
corre_table = pd.concat([faith_scores, temporal_distance, corpus_simi, text_length])
corre_table['Factors'] = index_faithful + index_time + index_corpus_simi + ['text avg length']
corre_table['Task'] = str(args.dataset)
corre_table.to_csv(datasets_dir + 'corre_table_' + str(similarity_method) + '.csv')


print(datasets_dir + 'corre_table_' + str(similarity_method) + '.csv')

