#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import pandas as pd
import argparse
import json
import logging
import gc
import datetime
import sys

# torch.cuda.empty_cache()
# # torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "yelp",
    # choices = ["WS", "SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr", "fc1", "fc2", "fc3"]
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
    # default = "full_text_models/",
    default="models/"
)


parser.add_argument(
    "--evaluation_dir",   
    type = str, 
    help = "directory to save faithfulness results", 
    default = "posthoc_results/"
)

parser.add_argument(
    "--extracted_rationale_dir",   
    type = str, 
    help = "directory to save extracted_rationales", 
    default = "extracted_rationales/"
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = None, 
    choices = [None, "kuma", "rl"]
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

log_dir = "experiment_logs/evaluate_" + user_args["dataset"] + "_" +  date_time + "/"
config_dir = "experiment_config/evaluate_" + user_args["dataset"] + "_" +  date_time + "/"


os.makedirs(log_dir, exist_ok = True)
os.makedirs(config_dir, exist_ok = True)


import config.cfg

config.cfg.config_directory = config_dir

logging.basicConfig(
                    filename= log_dir + "/out.log", 
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S'
                  )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1

logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()
CUDA_LAUNCH_BLOCKING=1

from src.common_code.initialiser import initial_preparations



# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "evaluate")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")








from src.data_functions.dataholders import BERT_HOLDER 
from src.evaluation import evaluation_pipeline

data = BERT_HOLDER(
    args["data_dir"], 
    stage = "eval",
    b_size = 4,
    #b_size = args["batch_size"], # TO FIX CUDA OUT OF MEMORY, MAY NOT WORK
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)
# will generate
logging.info("*********conducting in-domain flip experiments")
print('"*********conducting flip experiments on in-domain"')
evaluator.faithfulness_experiments_(data)
print('"********* DONE flip experiments on in-domain"')

del data
del evaluator
gc.collect()




## ood evaluation DATASET 1
data = BERT_HOLDER(
    path = args["data_dir"], 
    b_size = 4,
    #b_size = args["batch_size"], # TO FIX CUDA OUT OF MEMORY, MAY NOT WORK
    stage = "eval", #args["batch_size"],
    ood = True,
    ood_dataset_ = 1
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels,
    ood = True,
    ood_dataset_ = 1
)

logging.info("*********conducting oo-domain flip experiments DATASET 1")
print('"*********conducting oo-domain flip experiments DATASET 1"')
evaluator.faithfulness_experiments_(data)
print('"********* DONE oo-domain flip experiments DATASET 1"')

# delete full data not needed anymore
del data
del evaluator
gc.collect()




## ood evaluation DATASET 2
data = BERT_HOLDER(
    path = args["data_dir"], 
    b_size = 4,
    #b_size = args["batch_size"], # TO FIX CUDA OUT OF MEMORY, MAY NOT WORK
    stage = "eval", #args["batch_size"],
    ood = True,
    ood_dataset_ = 2
)

evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels,
    ood = True,
    ood_dataset_ = 2
)

logging.info("*********conducting oo-domain flip experiments DATASET 2")
print('"*********conducting oo-domain flip experiments DATASET 2"')

evaluator.faithfulness_experiments_(data)
print('"********* DONE oo-domain flip experiments DATASET 2"')
# delete full data not needed anymore
del data
del evaluator
gc.collect()
torch.cuda.empty_cache()