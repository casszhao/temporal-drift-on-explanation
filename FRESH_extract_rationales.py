#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import argparse
import logging
import gc
torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import datetime
import os


date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst", 
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
    default = "models/"
)

parser.add_argument(
    "--extracted_rationale_dir",   
    type = str, 
    help = "directory to save extracted_rationales", 
    default = "extracted_rationales/"
)

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = None, 
    choices = [None]
)

user_args = vars(parser.parse_args())

log_dir = "experiment_logs/extract_" + user_args["dataset"] + "_" +  date_time + "/"
config_dir = "experiment_config/extract_" + user_args["dataset"] + "_" + date_time + "/"


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

logging.info("Running on cuda ? {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.common_code.initialiser import initial_preparations
# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "extract")
print(args)
print('DONE initial preparations for args')


from src.evaluation import evaluation_pipeline
import datetime

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")



from src.data_functions.dataholders import BERT_HOLDER as dataholder



data = dataholder(
    args["data_dir"],
    b_size = 8, # b_size = args["batch_size"], #stage = "eval",
    return_as_frames = True
)



evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"], 
    output_dims = data.nu_of_labels
)


logging.info("*********extracting in-domain rationales")

evaluator.register_importance_(data, data_split='test')
evaluator.create_rationales_(data)

del data
del evaluator
gc.collect()
torch.cuda.empty_cache()
## ood evaluation DATASET 1
data = dataholder(
    path = args["data_dir"],
    b_size=8, # b_size = args["batch_size"],
    ood = True,
    ood_dataset_ = 1,
    return_as_frames = True
)
# data = dataholder(
#     path = args["data_dir"],
#     b_size = args["batch_size"],
#     ood = True,
#     ood_dataset_ = 1,
#     stage = "eval",
#     return_as_frames = True
# )


evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"],
    output_dims = data.nu_of_labels,
    ood = True,
    ood_dataset_ = 1
)

logging.info("*********extracting oo-domain rationales")

evaluator.register_importance_(data)
evaluator.create_rationales_(data)

# delete full data not needed anymore
del data
del evaluator
gc.collect()
torch.cuda.empty_cache()
## ood evaluation DATASET 2
data = dataholder(
    path = args["data_dir"],
    b_size=8, # b_size = args["batch_size"],
    ood = True,
    ood_dataset_ = 2,
    return_as_frames = True
)
# data = dataholder(
#     path = args["data_dir"],
#     # b_size = args["batch_size"],
#     b_size=16,
#     ood = True,
#     ood_dataset_ = 2,
#     stage = "eval",
#     return_as_frames = True
# )
evaluator = evaluation_pipeline.evaluate(
    model_path = args["model_dir"],
    output_dims = data.nu_of_labels,
    ood = True,
    ood_dataset_ = 2
)

logging.info("*********extracting oo-domain rationales")

evaluator.register_importance_(data) # register importance scores ()
evaluator.create_rationales_(data) # create json for training fresh ()

# delete full data not needed anymore
del data
del evaluator
gc.collect()
torch.cuda.empty_cache()

