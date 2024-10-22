
import torch
import pandas as pd
import json 
import glob 
import os
import logging
from src.common_code.initialiser import initial_preparations
from src.models.deterministic.tasc import lin as tasc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict, config_directory


# cwd = os.getcwd()  # Get the current working directory (cwd)
# # files = os.listdir(cwd)  # Get all the files in that directory
# # print("Files in %r: %s" % (cwd, files))


# os.makedirs(config.cfg.config_directory + 'instance_config.json', exist_ok = True)

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.deterministic.bert import BertClassifier
from src.evaluation.experiments.rationale_extractor import extract_importance_, rationale_creator_, extract_lime_scores_, extract_deeplift_values_, extract_gradientshap_values_, extract_deepliftshap_values_
from src.evaluation.experiments.erasure_tests import conduct_experiments_


import re

class evaluate():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, model_path, output_dims = 2, ood = False, ood_dataset_ = 0):
        
        """
        loads and holds a pretrained model
        """
        print(model_path, args["model_abbreviation"])
        self.models = glob.glob(model_path + args["model_abbreviation"] + "*.pt")
        self.output_dims = output_dims
        self.ood = ood
        ood_name = None
        if self.ood:

            assert ood_dataset_ in [1,2], (
                f"""
                Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                """
            )

            ood_name = args.ood_dataset_1 if ood_dataset_ == 1 else args.ood_dataset_2
        
        self.ood_dataset_ = ood_name

        logging.info(f" *** there are {len(self.models)} models in :  {model_path}")

        if len(self.models) == 0:

            raise FileNotFoundError(
                f"*** no models in directory -> {model_path}"
            )

    def register_importance_(self, data, data_split = None, model = None):
    
        for model_name in self.models:
            
            if args.use_tasc:
            
                tasc_variant = tasc
                
                tasc_mech = tasc_variant(data.vocab_size)
                
            else:
                
                tasc_mech = None

            model = BertClassifier(
                output_dim = self.output_dims,
                tasc = tasc_mech
            )

            logging.info(f" *** loading model -> {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model -> {model_name}")

            self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])


            for data_split_name, data_split in {"dev":  data.dev_loader, "train": data.train_loader, 
                                                "test":  data.test_loader}.items():
            #for data_split_name, data_split in {"test": data.test_loader}.items(): ## REMOVE AFTER
            # need to run extract importance first
                print(' ++++++++++++ START extracting for ', data_split_name, ' ++++++++++++++')

                
                extract_deeplift_values_(
                    model = model,
                    data = data_split,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    ood = self.ood,
                    ood_dataset_ = self.ood_dataset_
                )
                torch.cuda.empty_cache()
                print(' \\ +++++++++ DONE {} s deeplift '.format(data_split_name))
                
                
                extract_importance_(
                    model = model,
                    data_split_name = data_split_name,
                    data = data_split,
                    model_random_seed = self.model_random_seed,
                    ood = self.ood,
                    ood_dataset_ = self.ood_dataset_
                )
                torch.cuda.empty_cache()
                print(' \\ +++++++++ DONE {} s attributes of ig/scale attention/attention...'.format(data_split_name))


                extract_deepliftshap_values_(
                    model = model,
                    data = data_split,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    ood = self.ood,
                    ood_dataset_ = self.ood_dataset_
                )
                torch.cuda.empty_cache()
                print(' \\ +++++++++ DONE {} s deepliftshap '.format(data_split_name))


                extract_gradientshap_values_(
                    model = model,
                    data = data_split,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    ood = self.ood,
                    ood_dataset_ = self.ood_dataset_
                )
                torch.cuda.empty_cache()
                print(' \\ +++++++++ DONE {} s gradientshap '.format(data_split_name))

                extract_lime_scores_(
                    model = model,
                    data = data_split,
                    data_split_name = data_split_name,
                    model_random_seed = self.model_random_seed,
                    no_of_labels = data.nu_of_labels,
                    max_seq_len = data.max_len,
                    tokenizer = data.tokenizer,
                    ood = self.ood,
                    ood_dataset_ = self.ood_dataset_
                )
                
                torch.cuda.empty_cache()
                print(' \\ +++++++++ DONE {} s lime '.format(data_split_name))


        return 

    def create_rationales_(self, data):
        
        ## lets check how many models extracted importance_scores
        fname = os.path.join(
            os.getcwd(),
            args["extracted_rationale_dir"],
            "importance_scores",
            ""
        )

        for data_split_name, data_split in data.as_dataframes_().items():

            # if data_split_name in ["train", "dev"]: ## REMOVE AFTER to testing
            #
            #     continue

            score_list = glob.glob(fname + f"{data_split_name}*scores-*.npy")

            if args.use_tasc: score_list = [x for x in score_list if "tasc" in x]
            else: score_list = [x for x in score_list if "tasc" not in x]

            if self.ood: score_list = [x for x in score_list if f"-OOD-{self.ood_dataset_}-" in x]
            else: score_list = [x for x in score_list if f"-OOD-" not in x]
            
            model_seeds = [x.split(".npy")[0].split("-")[-1] for x in score_list]
            
            if len(model_seeds) > 1:
                print(' model seeds includes: -------')
                print(model_seeds)

                raise NotImplementedError("""

                Not yet implemented for more than one seeds for rationale extraction.
                Too expensive to run models on all rationales (e.g. 5 seeds x 6 feature scorings x 3 runs on each for training). 
                So just use one.
                """)

            for seed in model_seeds:

                rationale_creator_(
                    data = data_split,
                    data_split_name = data_split_name,
                    ood = self.ood,
                    ood_dataset_ = self.ood_dataset_,
                    tokenizer = data.tokenizer,
                    model_random_seed=seed
                )


        return

    def faithfulness_experiments_(self, data):
        
        for model_name in self.models:

            if args.use_tasc:
            
                tasc_variant = tasc
                
                tasc_mech = tasc_variant(data.vocab_size)
                
            else:
                
                tasc_mech = None

            model = BertClassifier(
                output_dim = self.output_dims,
                tasc = tasc_mech
            )

            logging.info(f" *** loading model - {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model - {model_name}")
            print('successfully loaded model for evaluating faithfulness')

            model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1])

            ## register importance scores if they are not there
            self.register_importance_(
                data, 
                #data_split = "test"
            ) 

            conduct_experiments_(
                model = model, 
                data = data.test_loader,
                model_random_seed = model_random_seed,
                ood = self.ood,
                ood_dataset_ = self.ood_dataset_
            )

        return