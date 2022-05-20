#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os 
import argparse
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_VISIBLE_DEVICES = 0
print(device)

import datetime
import gc

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "yelp_full",
    # choices = ["WS", "SST", "IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr", "factcheck","factcheck_ood2","factcheck_ood1"]
)

parser.add_argument(
    "--data_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)

parser.add_argument(
    "--model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "ft_model/"
)

parser.add_argument(
    "--seed",   
    type = int, 
    help = "random seed for experiment",
    default = 412
)

parser.add_argument(
    '--evaluate_models', 
    help='test predictive performance in and out of domain', 
    action='store_true'
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = "full_lstm", 
    choices = [None, "kuma", "rl", "full_lstm"]
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

log_dir = "ft_lstm_/" + user_args["dataset"] + "/ft_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_lstm" + date_time + "/"
config_dir = "experiment_config/train_" + user_args["dataset"] + "_seed-" + str(
    user_args["seed"]) + "_" + date_time + "/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)

import config.cfg

config.cfg.config_directory = config_dir

logging.basicConfig(
    filename=log_dir + "/lstm_out.log",
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
    stage = "train"
)

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")


if args["inherently_faithful"] is not None:
    
    from src.data_functions.dataholders import KUMA_RL_HOLDER as dataholder
    
else:
    
    from src.data_functions.dataholders import BERT_HOLDER as dataholder
    
from src.tRpipeline import train_and_save, test_predictive_performance, keep_best_model_

# training the models and evaluating their predictive performance
# on the full text length




logging.info(date_time)
logging.info("Finetune BERT for: {}".format(str(user_args["dataset"])))
logging.info("Finetune BERT for: {}".format(str(user_args["dataset"])))



batch_size_list = [8,16,32,64] #
for b in batch_size_list:
    data = dataholder(path=args["data_dir"], b_size=b)
    logging.info(" \\ ------------------  batch size: {}".format(str(b)))

    #LR = [1e-4, 5e-4, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-6, 5e-6]
    LR = [1e-2, 3e-2, 5e-2, 1e-3, 3e-3, 5e-3, 1e-4, 3e-4, 5e-4, 1e-5, 3e-5, 5e-5]
    for lr in LR:
        logging.info(" \\ -------------------- learning rate: {}".format(lr))
        train_searchPara_and_save(
            train_data_loader=data.train_loader,
            dev_data_loader=data.dev_loader,
            output_dims=data.nu_of_labels,
            lr = lr, #3e-5, 2e-5
        )

del data
gc.collect()