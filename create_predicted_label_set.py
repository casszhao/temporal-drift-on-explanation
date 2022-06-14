import json
import numpy as np


# change indomain



def change_label_and_create_new_df(model_output_array, extracted_rationales_path, new_save_path):  

    model_output_array = np.load(model_output_array, allow_pickle= True).item()  

    with open(extracted_rationales_path) as file:
        data = json.load(file)

    for doc in data:
        docid = doc['annotation_id']
        predicted =  model_output_array[docid]['predicted'].argmax()
        doc['label'] = int(predicted)

    with open(new_save_path, 'w') as file:
        json.dump(
            data,
            file,
            indent = 4)



def generate_for_one_task_one_feature(data, features, seed):
    model_output_path_ind: str = f'models/{data}/bert-output_seed-{seed}.npy'
    rationales_path_ind: str = f'extracted_rationales/{data}/data/topk/{features}-test.json'
    new_data_path_ind: str = f'datasets_{feature}/{data}/data/'

    model_output_path_ood1: str = f'models/{data}/bert-output_seed-{seed}-OOD-{data}_ood1.npy'
    rationales_path_ood1: str = f'extracted_rationales/{data}/data/topk/OOD-{data}_ood1-{features}-test.json'
    new_data_path_ood1: str = f'datasets_{feature}/{data}_ood1/data/'

    model_output_path_ood2: str = f'models/{data}/bert-output_seed-{seed}-OOD-{data}_ood2.npy'
    rationales_path_ood2: str = f'extracted_rationales/{data}/data/topk/OOD-{data}_ood2-{features}-test.json'
    new_data_path_ood2: str = f'datasets_{feature}/{data}_ood2/data/'

    import os

    for path in (
                    new_data_path_ind, 
                    new_data_path_ood1,
                    new_data_path_ood2
                ):

        os.makedirs(path, exist_ok=True)


    change_label_and_create_new_df(model_output_path_ind, rationales_path_ind, new_data_path_ind)
    change_label_and_create_new_df(model_output_path_ood1, rationales_path_ood1, new_data_path_ood1)
    change_label_and_create_new_df(model_output_path_ood2, rationales_path_ood2, new_data_path_ood2)

# can do task list in batch
# can only do feature by feature
# data = 'factcheck' # yelp 25 / agnews 25 / xfact 5 / factcheck 5 / AmazDigiMu 20 / AmazPantry 15
# feature = 'gradients' #scaled attention # deeplift # gradients 
# seed = 5

import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "yelp", 
)

parser.add_argument(
    "--feature", 
    type = str, 
    help = "select dataset / task", 
    default = "yelp", 
)

arguments = vars(parser.parse_args())

seeds = {
    "factcheck" : 5,
    "yelp" : 25,
    "agnews" : 25,
    "xfact" : 5,
    "factcheck" : 5,
    "AmazDigiMu" : 20,
    "AmaziPantry" : 20
}

data = arguments["dataset"]
seed = seeds[data]
feature = arguments["feature"]

generate_for_one_task_one_feature(data, feature, seed)