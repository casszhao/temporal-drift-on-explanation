# train on all dataset --> fine-tune hyperparemeters to get the best

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import argparse
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import datetime
import gc

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    help="select dataset / task",
    default="yelp",
    # choices = ["WS", "SST", "IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr", "factcheck","factcheck_ood2","factcheck_ood1"]
)

parser.add_argument(
    "--data_dir",
    type=str,
    help="directory of saved processed data",
    default="datasets/"
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="directory to save models",
    default="ft_kuma/"
)

parser.add_argument(
    "--seed",
    type=int,
    help="random seed for experiment",
    default= 412
)

parser.add_argument(
    '--evaluate_models',
    help='test predictive performance in and out of domain',
    action='store_true'
)

parser.add_argument(
    "--inherently_faithful",
    type=str,
    help="select dataset / task",
    default="kuma",
    choices=[None, "kuma", "rl", "full_lstm"]
)

parser.add_argument(
    '--use_tasc',
    help='for using the component by GChrys and Aletras 2021',
    action='store_true'
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

log_dir = "ft_kuma/" + user_args["dataset"] + "/ft_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_bert" + date_time + "/"
config_dir = "ft_kuma/train_" + user_args["dataset"] + "_seed-" + str(
    user_args["seed"]) + "_" + date_time + "/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

import config.cfg

config.cfg.config_directory = config_dir

logging.basicConfig(
    filename=log_dir + "/ft_kuma.log",
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.info("Running on cuda : {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.common_code.initialiser import initial_preparations
import datetime

# creating unique config from user args and model_config.json file
args = initial_preparations(
    user_args,
    stage="train"
)

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k, v in args.items()]
logging.info("\n ----------------------")

if args["inherently_faithful"] is not None:

    from src.data_functions.dataholders import KUMA_RL_HOLDER as dataholder

else:

    from src.data_functions.dataholders import BERT_HOLDER as dataholder

from src.tRpipeline import train_and_save, test_predictive_performance, keep_best_model_, train_searchPara_and_save

# training the models and evaluating their predictive performance
# on the full text length

data = dataholder(
    path=args["data_dir"],
    b_size=8)

## evaluating finetuned models
if args["evaluate_models"]:

    ## in domain evaluation
    test_stats = test_predictive_performance(
        test_data_loader=data.test_loader,
        for_rationale=False,
        output_dims=data.nu_of_labels,
        save_output_probs=True,
        vocab_size=data.vocab_size
    )

    del data
    gc.collect()


    ## shows which model performed best on dev F1 (in-domain)
    ## if keep_models = False then will remove the rest of the models to save space
    print(' ---- keep_models = False, will remove the rest of the models to save space')
    keep_best_model_(keep_models=False)

else:
    logging.info(date_time)
    logging.info("Finetune BERT for: {}".format(str(user_args["dataset"])))
    logging.info("Finetune BERT for: {}".format(str(user_args["dataset"])))
    

    
    train_and_save(
        train_data_loader = data.train_loader, 
        dev_data_loader = data.dev_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        vocab_size = data.vocab_size
    )


import numpy as np
import logging
import os
import argparse


dataset = str(user_args["dataset"])


logging.info(f'''
        {dataset} ----''')



def one_domain_len(domain):
    overall = np.zeros(1)
    path_to_file : str = f'ft_kuma/{dataset}/kuma-bert-output_seed-None.npy'
        
    file_data = np.load(path_to_file, allow_pickle=True).item()
    #print(file_data)
    aggregated_ratio = np.zeros(len(file_data))

    for _i_, (docid, metadata) in enumerate(file_data.items()):

        rationale_ratio = min( 
            1.,
            metadata['rationale'].sum()/metadata['full text length']
        )

        aggregated_ratio[_i_] = rationale_ratio

    overall = aggregated_ratio.mean()
    
    logging.info(f'''
        {domain}
        mean -> {overall.mean()}
        std ->  {overall.std()}
        all ->  {overall}
    ''')

    print(f'''{domain}
        mean -> {overall.mean()}
        std ->  {overall.std()}
        all ->  {overall}
    ''')


one_domain_len('InDomain')