import urllib.request
import os
import nltk
import spacy
import zipfile
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from cleaners_encoders import cleaner, tokenize, invert_and_join
import tarfile
import argparse
import csv
import sys
import shutil 
import pandas as pd
from sklearn.utils import shuffle

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_directory', 
    type=str, 
    help='directory to save processed data', 
    default = "datasets"
)

args = parser.parse_args()

AMAZ_DATA_ = {
    "AmazDigiMu": "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music_5.json.gz",
    "AmazPantry" : "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry_5.json.gz",
    "AmazInstr" : "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments_5.json.gz"
}

import gc

TEMP_DATA_DIR=".temp_data/"

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])

def download_raw_data():

    ## create our temp data directory
    os.makedirs(TEMP_DATA_DIR, exist_ok = True)

    """
    Amazon datasets retrieved from Retrieved from 
    https://nijianmo.github.io/amazon/index.html#complete-data
    Justifying recommendations using distantly-labeled reviews and fined-grained aspects
    Jianmo Ni, Jiacheng Li, Julian McAuley
    Empirical Methods in Natural Language Processing (EMNLP), 2019
    """

    for dataset, url in AMAZ_DATA_.items():

        name = url.split("/")[-1]
        fname = f"{TEMP_DATA_DIR}{name}"

        print(f"*** downloading and extracting data for {dataset}")
        if os.path.exists(fname):
            print(f"**** {TEMP_DATA_DIR}{name} allready exists")
            pass

        else:

            urllib.request.urlretrieve(
                url, 
                f"{TEMP_DATA_DIR}{name}"
            )
# ''''''
#     ### YELP
#     fname = f"{TEMP_DATA_DIR}yelp_review_polarity_csv.tgz"
#
#     print("*** downloading and extracting data for  YELP")
#     if os.path.exists(fname):
#         print(f"**** {TEMP_DATA_DIR}yelp_review_polarity_csv.tgz allready exists")
#         pass
#
#     else:
#
#         # download raw for MNLI, QNLI, QQP, TwitterPPDB, SWAG, HELLASWAG
#         ## file_id from Desai & Durret, Calibration of Pre-trained Transformers github page
#         urllib.request.urlretrieve(
#             "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz",
#             f"{TEMP_DATA_DIR}yelp_review_polarity_csv.tgz"
#         )
#
#         if fname.endswith("tgz"):
#
#             tar = tarfile.open(fname, "r:gz")
#             tar.extractall(path = TEMP_DATA_DIR)
#             tar.close()
#
#     ## download data for IMDB
#     # Download the files from `url` and save it locally under `file_name`
#     print("*** downloading and extracting data for IMDB")
#     if os.path.exists(f"{TEMP_DATA_DIR}imdb_full.pkl"):
#
#         print(f"*** {TEMP_DATA_DIR}imdb_full.pkl allready exists")
#         pass
#
#     else:
#
#         urllib.request.urlretrieve(
#             "https://s3.amazonaws.com/text-datasets/imdb_full.pkl",
#             f"{TEMP_DATA_DIR}imdb_full.pkl"
#         )
#
#     if os.path.exists(f"{TEMP_DATA_DIR}imdb_word_index.json"):
#
#         print(f"*** {TEMP_DATA_DIR}imdb_word_index.json allready exists")
#
#         pass
#
#     else:
#
#         urllib.request.urlretrieve(
#             "https://s3.amazonaws.com/text-datasets/imdb_word_index.json",
#             f"{TEMP_DATA_DIR}imdb_word_index.json"
#         )
#
#     ## Download raw for SST
#     # Download the file from `url` and save it locally under `file_name`
#     print("*** downloading and extracting data for SST")
#     if os.path.exists(f"{TEMP_DATA_DIR}sst_data.zip"):
#
#         print(f"*** {TEMP_DATA_DIR}sst_data.zip allready exists")
#
#         pass
#
#     else:
#
#         urllib.request.urlretrieve(
#             "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
#             f"{TEMP_DATA_DIR}sst_data.zip"
#         )
# ''''''
#         # extract files
#         with zipfile.ZipFile(f"{TEMP_DATA_DIR}sst_data.zip", 'r') as zip_ref:
#             zip_ref.extractall(TEMP_DATA_DIR)

    print("*** downloaded and extracted all temp files succesfully")

    return

class SSTProcessor():

    """Processor for SST"""

    def load_samples_(self, path_to_data):

        a = nltk.corpus.BracketParseCorpusReader(f"{TEMP_DATA_DIR}trees/", "(train|dev|test)\.txt")

        text = {}
        labels = {}
        label_ids = {}
        annotation_ids = {}
        keys = ['train', 'dev', 'test']

        for split in keys :
            ## parse text
            text[split] = [x.leaves() for x in a.parsed_sents(split+'.txt') if x.label() != '2']
            ## tokenize text
            text[split] = [tokenize(t) for t in text[split]]

            ## prepare labels
            labels[split] = [int(x.label()) for x in a.parsed_sents(split+'.txt') if x.label() != '2']
            labels[split] = [1 if x >= 3 else 0 for x in labels[split]]

            ## label ids for better clarity
            label_ids[split] = ["positive" if x == 1 else "negative" for x in labels[split]]

            ## place unique identifiers
            annotation_ids[split] = [f"{split}_{r}" for r in range(len(text[split]))]

            dataset = []

            for _i_ in range(len(text[split])):
                
                dataset.append({
                    "annotation_id" : annotation_ids[split][_i_],
                    "exp_split" : split,
                    "text" : text[split][_i_],
                    "label" : labels[split][_i_],
                    "label_id" : label_ids[split][_i_]
                })

            print(f"{split} -> {len(text[split])}")

            ## save our dataset
            os.makedirs(path_to_data, exist_ok = True)

            with open(path_to_data + f"{split}.json", "w") as file:

                json.dump(
                    dataset,
                    file,
                    indent = 4
                )

        return

class IMDBProcessor():

    """Processor for IMDB"""

    def load_samples_(self, path_to_data):

        data = pickle.load(open(f"{TEMP_DATA_DIR}imdb_full.pkl", "rb"))
        vocab = json.load(open(f"{TEMP_DATA_DIR}imdb_word_index.json"))

        inv = {idx:word for word, idx in vocab.items()}

        (X_train, y_train), (Xt, yt) = data

        trainidx = np.arange(len(X_train)) 

        trainidx, testdevidx = train_test_split(trainidx, train_size=0.8, random_state=1378)
        devidx, testidx =  train_test_split(testdevidx, train_size=0.5, random_state=1378)

        for split , indxs in {"train" : trainidx, "dev" : devidx, "test" : testidx}.items():
            
            X = [X_train[i] for i in indxs]
            y = [y_train[i] for i in indxs]

            X = invert_and_join(X, idx_to_word = inv)
            
            dataset = []

            for _i_ in tqdm(range(len(X)), desc = f"registering for -> {split}"):

                dataset.append({
                    "annotation_id" : f"{split}_{_i_}",
                    "exp_split" : split,
                    "text" : " ".join(cleaner(X[_i_])),
                    "label" : y[_i_],
                    "label_id" : "positive" if  y[_i_] == 1 else "negative"
                })
            
            print(f"{split} -> {len(y)}")

            ## save our dataset
            os.makedirs(path_to_data, exist_ok = True)

            with open(path_to_data + f"{split}.json", "w") as file:
                json.dump(
                    dataset,
                    file,
                    indent = 4
                )

        return

class YelpProcessor():

    """Processor for IMDB"""

    def load_samples_(self, path_to_data):

        df = pd.read_csv(f"{TEMP_DATA_DIR}yelp_review_polarity_csv/train.csv", header = None)
        df = df.rename(columns = {0:"label", 1:"text"})
        from sklearn.model_selection import train_test_split
        X_train, X_dev, y_train, y_dev = train_test_split(
            df["text"], 
            df["label"], 
            stratify=df["label"], 
            test_size=0.15,
            random_state = 18
        )


        df = pd.read_csv(f"{TEMP_DATA_DIR}yelp_review_polarity_csv/test.csv", header = None)
        df = df.rename(columns = {0:"label", 1:"text"})
        X_test, y_test = df["text"], df["label"]


        for split , (X,y) in {"train" : (X_train, y_train), "dev" : (X_dev,y_dev), "test" : (X_test, y_test)}.items():
            
            dataset = []
            
            X, y = X.values, y.values
            
            for _i_ in tqdm(range(len(X)), desc = f"registering for -> {split}"):
                
                dataset.append({
                    "annotation_id" : f"{split}_{_i_}",
                    "exp_split" : split,
                    "text" : " ".join(cleaner(X[_i_].replace("\\n", ""))),
                    "label" : 1 if y[_i_] == 2 else 0,
                    "label_id" : "positive" if  y[_i_] == 2 else "negative"
                })

            print(f"{split} -> {len(y)}")

            ## save our dataset
            os.makedirs(path_to_data, exist_ok = True)

            with open(path_to_data + f"{split}.json", "w") as file:
                json.dump(
                    dataset,
                    file,
                    indent = 4
                )
        
        
        del df
        gc.collect()

        return

import gzip

class ProcessAmazonDatasets():

    """Processor for All Amazon datasets"""

    def load_samples_(self, task_name, path_to_data):

        sent_lab = {2: "positive", 1: "neutral", 0:"negative"}

        data = []
        counter = 0

        print(' now processing: ', str(AMAZ_DATA_[task_name]))

        file_loc = os.path.join(
            os.getcwd(),
            TEMP_DATA_DIR,
            AMAZ_DATA_[task_name].split("/")[-1]
        )

        with gzip.open(file_loc, "rb") as file: 
            
            for line in tqdm(file.readlines()):
                
                try:
                    
                    json_acceptable_string = line.decode("utf-8").replace("\"", "\"").rstrip()
                    d = json.loads(json_acceptable_string)
                    score = d["overall"]
                    date = d["reviewTime"]
                    verified = d["verified"]
                    label = self.score2sent(score)
                    data.append({
                        "text" : " ".join(cleaner(d["reviewText"])),
                        "true score" : score,
                        "label" : label,
                        "label_id" : sent_lab[label],
                        "date":  date,
                        "verified": verified
                    })

                except KeyError as e: 
                    
                    counter += 1
            
        print(f"*** failed to convert {counter} instances. This is due to KeyError (i.e. no review text found)")

        df = pd.DataFrame(data)

        train_indx, testdev_indx = train_test_split(df.index, test_size=0.2, stratify=df["label"])
        train = df.iloc[train_indx]
        testdev = df.loc[testdev_indx, :]
        train["split"] = "train"

        assert len([x for x in testdev.index if x in train.index]) == 0, ("""
        data leakage
        """)

        test_indx, dev_indx = train_test_split(testdev.index, test_size=0.5, stratify=testdev["label"])
        test = df.loc[test_indx, :]
        test["split"] = "test"
        dev = df.loc[dev_indx, :]
        dev["split"] = "dev"

        assert len([x for x in dev.index if x in test.index]) == 0, ("""
        data leakage
        """)

        train.reset_index(inplace = True)
        train["annotation_id"] = train.apply(lambda row: "train_" + str(row.name), axis = 1)

        dev.reset_index(inplace = True)
        dev["annotation_id"] = dev.apply(lambda row: "dev_" + str(row.name), axis = 1)

        test.reset_index(inplace = True)
        test["annotation_id"] = test.apply(lambda row: "test_" + str(row.name), axis = 1)



        for split, data in {"train": train, "dev": dev, "test":test}.items():

            ## save our dataset
            os.makedirs(path_to_data, exist_ok = True)

            with open(path_to_data + f"{split}.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4
                )

        ### sorted dataset

        df_sorted = df.copy()
        df_sorted['date'] = pd.to_datetime(df_sorted['date']).dt.date
        df_sorted = df.sort_values(by='date', na_position='first')
        ood1_len = ood2_len = indomain_test_len = indomain_dev_len = int(len(df_sorted) * 0.1)
        in_domain_len = len(df_sorted) - ood1_len * 2
        assert len(df_sorted) == in_domain_len + ood2_len + ood1_len
        print('full in domain length: ', in_domain_len)
        print('ood length: ', ood1_len)

        in_domain = df_sorted.iloc[:in_domain_len]
        in_domain = shuffle(in_domain)
        ood1 = df_sorted.iloc[in_domain_len:in_domain_len + ood1_len]
        ood2 = df_sorted.iloc[in_domain_len + ood1_len:]

        in_domain_train, in_domain_test = train_test_split(in_domain, train_size=0.75, stratify=in_domain['label'])
        in_domain_dev, in_domain_test = train_test_split(in_domain_test, train_size=0.5,
                                                         stratify=in_domain_test['label'])

        for split, data in {"train": in_domain_train, "test": in_domain_test, "dev":in_domain_dev}.items():

            ## save our dataset
            os.makedirs(path_to_data + "in_domain/data/", exist_ok = True)

            with open(path_to_data + "in_domain/data/" + f"{split}.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4
                )

        for split, data in {"ood1": ood1, "ood2": ood2}.items():

            ## save our dataset

            ood_data_directory = os.path.join(
                os.getcwd(),
                args.data_directory,
                task_name,
                str(split),
                "data",
                ""
            )
            os.makedirs(ood_data_directory, exist_ok=True)

            with open(ood_data_directory + "test.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4
                )
            with open(ood_data_directory + "train.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4
                )
            with open(ood_data_directory + "dev.json", "w") as file:
                json.dump(
                    data.to_dict("records"),
                    file,
                    indent = 4
                )

        return


    def score2sent(self, score : float) -> int:
    
        if score > 3:
            return 2
        if score < 3:
            return 0    
        return 1


class AmazDigiMuProcessor(ProcessAmazonDatasets):

    def __init__(self):

        return

class AmazInstrProcessor(ProcessAmazonDatasets):

    def __init__(self):

        return

class AmazPantryProcessor(ProcessAmazonDatasets):

    def __init__(self):

        return



def describe_data_stats(path_to_data, path_to_stats):
    """ 
    returns dataset statistics such as : 
                                        - number of documens
                                        - average sequence length
                                        - average query length (if QA)
    """

    descriptions = {"train":{}, "dev":{}, "test":{}}
    
    for split_name in descriptions.keys():

        with open(f"{path_to_data}{split_name}.json", "r") as file: data = json.load(file)

        if "query" in data[0].keys(): 


            avg_ctx_len = np.asarray([len(x["document"].split(" ")) for x in data]).mean()
            avg_query_len = np.asarray([len(x["query"].split(" ")) for x in data]).mean()

            descriptions[split_name]["avg. context length"] = int(avg_ctx_len)
            descriptions[split_name]["avg. query length"] = int(avg_query_len)

        else:

            avg_seq_len = np.asarray([len(x["text"].split(" ")) for x in data]).mean()

            descriptions[split_name]["avg. sequence length"] = int(avg_seq_len)

        descriptions[split_name]["no. of documents"] = int(len(data))
        
        label_nos = np.unique(np.asarray([x["label"] for x in data]), return_counts = True)

        for label, no_of_docs in zip(label_nos[0], label_nos[1]):

            descriptions[split_name][f"docs in label-{label}"] = int(no_of_docs)
    
    ## save descriptors
    fname = path_to_stats + "dataset_statistics.json"

    with open(fname, 'w') as file:
        
            json.dump(
                descriptions,
                file,
                indent = 4
            ) 


    del data
    del descriptions
    gc.collect()

    return


if __name__ == "__main__":
    
    ## download the raw temp data
    if os.path.isdir(TEMP_DATA_DIR):

        pass

    else:

        download_raw_data()

    for task_name in {"AmazDigiMu", "AmazPantry", "AmazInstr"}: #"SST","IMDB", "Yelp",

        print(f"** processing -> {task_name}")

        processor = globals()[f'{task_name}Processor']()

        data_directory = os.path.join(
            os.getcwd(),
            args.data_directory,
            task_name, 
            "data",
            ""
        )
        
        if "Amaz" in task_name:
            
            dataset = processor.load_samples_(
                task_name = task_name,
                path_to_data = data_directory
            )

        else:

            dataset = processor.load_samples_(data_directory)

        ## save stats in 
        stats_directory = os.path.join(
            os.getcwd(),
            args.data_directory,
            task_name, 
            ""
        )

        describe_data_stats(
            path_to_data = data_directory,
            path_to_stats = stats_directory
        )

    print(f"** removing temporary data")

    # # deleting temp_data
    # shutil.rmtree(TEMP_DATA_DIR)
