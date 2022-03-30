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
# from robo.fmin import bayesian_optimization

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
    default = "complain",
)
args = parser.parse_args()

datasets_dir = 'saved_everything/' + str(args.dataset)
os.makedirs(datasets_dir, exist_ok = True)




'''
comparing
'''

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




def pre_post_process(domain_reps, domain_column_name):
    df = get_similarity_between_2reps(InD_train_reps, domain_reps, feature_names)
    df['Measure'] = Measure
    df['Rep_Mea'] = Rep_Mea
    df['Domain'] = str(domain_column_name)
    return df



Rep_Mea = ['Term jensen-shannon', 'Term renyi', 'Term cosine', 'Term euclidean', 'Term variational', 'Term bhattacharyya',
           'Topic jensen-shannon', 'Topic renyi', 'Topic cosine', 'Topic euclidean', 'Topic variational', 'Topic bhattacharyya']

Measure = ['jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya',
           'jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']


# data = 'factcheck'
feature_set_names = ['similarity', 'topic_similarity']
feature_names = features.get_feature_names(feature_set_names)
# for topic modelling:





###################################### 5. domain similarity between:  In domain / ood1 / ood2
#########################################################################################
num_iterations = 2000 # for testing, original use 2000? need to check the paper
VOCAB_SIZE = 20000
model_dir = 'similarity_models/full_text/'+ str(args.dataset) + '/vocab/' + str(VOCAB_SIZE)
os.makedirs(model_dir, exist_ok=True)

InD_train_list = pd.read_json('./datasets/'+str(args.dataset)+'/data/train.json')['text']
InD_test_list = pd.read_json('./datasets/'+str(args.dataset)+'/data/test.json')['text']
OOD1_list= pd.read_json('./datasets/'+str(args.dataset)+'_ood1/data/test.json')['text']
OOD2_list= pd.read_json('./datasets/'+str(args.dataset)+'_ood2/data/test.json')['text']

in_domain_train_list_list = convert_to_listoflisttoken(InD_train_list)
in_domain_test_list_list = convert_to_listoflisttoken(InD_test_list)
OOD1_test_list_list = convert_to_listoflisttoken(OOD1_list)
OOD2_test_list_list = convert_to_listoflisttoken(OOD2_list)

list_of_list_of_tokens = in_domain_train_list_list + in_domain_test_list_list + OOD1_test_list_list + OOD2_test_list_list

# create the vocabulary or load it if it was already created
vocab_path = os.path.join(model_dir, 'vocab.txt')
vocab = data_utils.Vocab(VOCAB_SIZE, vocab_path) # two functions, load and create
vocab.create(list_of_list_of_tokens, lowercase=True)

term_dist_path = os.path.join(datasets_dir, 'term_dist.txt')
topic_vectorizer, lda_model = similarity.train_topic_model(in_domain_train_list_list, vocab, num_topics=50, num_iterations=num_iterations, num_passes=10)

InD_train_reps = features.get_reps_for_one_domain(in_domain_train_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True) # 0. term dist 1. topic dist
InD_test_reps = features.get_reps_for_one_domain(in_domain_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)
OOD1_reps = features.get_reps_for_one_domain(OOD1_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)
OOD2_reps = features.get_reps_for_one_domain(OOD2_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)

baseline_similarity = pre_post_process(InD_test_reps, 'In Domain(Baseline)')
OOD1_similarity = pre_post_process(OOD1_reps, 'OOD1')
OOD2_similarity = pre_post_process(OOD2_reps, 'OOD2')
results = pd.concat([baseline_similarity,OOD1_similarity,OOD2_similarity],ignore_index=True)

results.to_csv(datasets_dir + '/fulltext_similarity_vocab' + str(vocab.size) + ' .csv')


################################################ 6. rationale similarity between:  In domain / ood1 / ood2
###############################################################################################################
num_iterations = 2000 # for testing, original use 2000? need to check the paper
VOCAB_SIZE = 20000
model_dir = 'similarity_models/for_rationales/'+ str(args.dataset) + '/vocab/' + str(VOCAB_SIZE)
os.makedirs(model_dir, exist_ok=True)


##### get rationales ####

thresh_list = []
for threshold in ['topk', 'contigious']:
    attributes_list = []
    for attribute_name in ['attention', 'lime', 'deeplift', 'gradients', 'scaled attention']:

        InD_path_train = './extracted_rationales/'+str(args.dataset)+'/data/'+str(threshold)+'/'+str(attribute_name)+'-train.json'
        InD_path_test = './extracted_rationales/'+str(args.dataset)+'/data/'+str(threshold)+'/'+str(attribute_name)+'-test.json'
        OOD1_path = 'extracted_rationales/'+str(args.dataset)+'/data/'+str(threshold)+'/OOD-'+str(args.dataset)+'_ood1-'+str(attribute_name)+'-test.json'
        OOD2_path = 'extracted_rationales/'+str(args.dataset)+'/data/'+str(threshold)+'/OOD-'+str(args.dataset)+'_ood2-'+str(attribute_name)+'-test.json'
        InD_train_list = pd.read_json(InD_path_train)['text']
        InD_test_list = pd.read_json(InD_path_test)['text']
        OOD1_list = pd.read_json(OOD1_path)['text']
        OOD2_list = pd.read_json(OOD2_path)['text']
        #########################

        in_domain_train_list_list = convert_to_listoflisttoken(InD_train_list)
        in_domain_test_list_list = convert_to_listoflisttoken(InD_test_list)
        OOD1_test_list_list = convert_to_listoflisttoken(OOD1_list)
        OOD2_test_list_list = convert_to_listoflisttoken(OOD2_list)

        list_of_list_of_tokens = in_domain_train_list_list + in_domain_test_list_list + OOD1_test_list_list + OOD2_test_list_list


        # create the vocabulary or load it if it was already created
        vocab_path = os.path.join(model_dir, 'vocab.txt')
        vocab = data_utils.Vocab(VOCAB_SIZE, vocab_path) # two functions, load and create
        vocab.create(list_of_list_of_tokens, lowercase=True)

        term_dist_path = os.path.join(datasets_dir, str(threshold), str(attribute_name), 'term_dist.txt')
        topic_vectorizer, lda_model = similarity.train_topic_model(in_domain_train_list_list, vocab, num_topics=50, num_iterations=num_iterations, num_passes=10)

        InD_train_reps = features.get_reps_for_one_domain(in_domain_train_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True) # 0. term dist 1. topic dist
        InD_test_reps = features.get_reps_for_one_domain(in_domain_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)
        OOD1_reps = features.get_reps_for_one_domain(OOD1_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)
        OOD2_reps = features.get_reps_for_one_domain(OOD2_test_list_list, vocab, feature_names, topic_vectorizer, lda_model, lowercase=True)

        baseline_similarity = pre_post_process(InD_test_reps, 'In Domain(Baseline)')
        OOD1_similarity = pre_post_process(OOD1_reps, 'OOD1')
        OOD2_similarity = pre_post_process(OOD2_reps, 'OOD2')
        results = pd.concat([baseline_similarity,OOD1_similarity,OOD2_similarity],ignore_index=True)

        results['attribute_name'] = str(attribute_name)

        attributes_list.append(results)
    all_attributes_df = pd.concat([attributes_list[0], attributes_list[1], attributes_list[2], attributes_list[3], attributes_list[4], attributes_list[5]], ignore_index=False)
    all_attributes_df['threshold'] = str(threshold)

    thresh_list.append(all_attributes_df)

results = pd.concat([thresh_list[0], thresh_list[1]], ignore_index=True)

results.to_csv(datasets_dir + '/rationale_similarity_vocab' + str(vocab.size) + ' .csv')




####################################### 7. datasets metadata: train/test/ size, time span, label distribution

