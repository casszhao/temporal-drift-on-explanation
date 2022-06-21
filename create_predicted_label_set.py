import json
import numpy as np
import pandas as pd
from turtle import color
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
import fnmatch
import os


# change indomain

# def change_label_and_create_new_df(model_output_array, extracted_rationales_path, new_save_path):  

#     model_output_array = np.load(model_output_array, allow_pickle= True).item()  

#     with open(extracted_rationales_path) as file:
#         data = json.load(file)

#     for doc in data:
#         docid = doc['annotation_id']
#         predicted =  model_output_array[docid]['predicted'].argmax()
#         doc['label'] = int(predicted)
    
#     print(data)

#     with open(new_save_path, 'w') as file:
#         json.dump(
#             data,
#             file,
#             indent = 4)



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

parser.add_argument(
    '--combine_and_get_f1',
    help='decide which parts are in need',
    action='store_true',
    default=True, 
)

args = parser.parse_args()

arguments = vars(parser.parse_args())

seeds = {
    "factcheck" : 5,
    "yelp" : 25,
    "agnews" : 25,
    "xfact" : 5,
    "factcheck" : 5,
    "AmazDigiMu" : 20,
    "AmazPantry" : 20
}

data = arguments["dataset"]
seed = seeds[data]
feature = arguments["feature"]



if args.combine_and_get_f1:
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 2, figsize=(4.8, 6.5), sharey=False, sharex=False)

    marker_style = dict(color='tab:blue', linestyle=':', marker='d',
                        #markersize=15, markerfacecoloralt='tab:red',
                        )
    xlabel_size = 12
    xtick_size = 22
    ytick_size = 22
    makersize = 66
    from sklearn.metrics import f1_score, accuracy_score

    task_list = ['agnews', 'xfact', 'factcheck', 'AmazDigiMu', 'AmazPantry', 'yelp']
    result_list = []
    for i, name in enumerate(task_list):
        full_results = pd.read_csv('./saved_everything/'+str(name)+'/fresh_bert_pred_compare.csv')
        if 'Amaz' in name:
            SUB_NAME = str(name)
        else:
            SUB_NAME = str(name).capitalize()
        scaled_attention_f1_list = []
        deeplift_f1_list = []
        gradients_f1_list = []
        domain_list = ['Full', 'SynD', 'AsyD1', 'AsyD2']
        for domain in domain_list:
            domain_df = full_results[full_results['Domain']==str(domain)]
            scaled_attention_f1 = f1_score(domain_df['bert_pred_label'], domain_df['scaled_pred_label'], average='macro')
            deeplift_f1 = f1_score(domain_df['bert_pred_label'], domain_df['deeplift_pred_label'], average='macro')
            gradients_f1 = f1_score(domain_df['bert_pred_label'], domain_df['gradients_pred_label'], average='macro')

            scaled_attention_f1_list.append(scaled_attention_f1*100)
            deeplift_f1_list.append(deeplift_f1*100)
            gradients_f1_list.append(gradients_f1*100)

        df = pd.DataFrame({'scaled attention':scaled_attention_f1_list, 'deeplift': deeplift_f1_list,
                            'gradients':gradients_f1_list, 'Domain':domain_list})
        print(df)

        if i < 2:
            axs[0, i].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', color='dimgrey') #, marker='x', s=makersize
            axs[0, i].scatter(df['Domain'], df['deeplift'], label='DL', marker='d', color='darkorange')
            axs[0, i].scatter(df['Domain'], df['scaled attention'], label=r'$\alpha\nabla\alpha$', marker='<', color='steelblue')
            #axs[0, i].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', color='red')
            axs[0, i].set_xlabel(SUB_NAME,fontsize=xlabel_size)
            axs[0, i].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
        elif 4 > i > 1:
            axs[1, i-2].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', color='dimgrey')
            axs[1, i-2].scatter(df['Domain'], df['deeplift'], label='DL', marker='d', color='darkorange')
            axs[1, i-2].scatter(df['Domain'], df['scaled attention'], label=r'$\alpha\nabla\alpha$', marker='<', color='steelblue')
            #axs[1, i-2].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', color='red')
            axs[1, i-2].set_xlabel(SUB_NAME,fontsize=xlabel_size)
            axs[1, i-2].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
        else:
            axs[2, i-4].scatter(df['Domain'], df['gradients'], label=r'$x\nabla x $', color='dimgrey')
            axs[2, i-4].scatter(df['Domain'], df['deeplift'], label='DL', marker='d', color='darkorange')
            axs[2, i-4].scatter(df['Domain'], df['scaled attention'], label=r'$\alpha\nabla\alpha$', marker='<', color='steelblue')
            #axs[2, i-4].scatter(df['Domain'], df['BERT'], label='BERT', marker='x', color='red')
            axs[2, i-4].set_xlabel(SUB_NAME,fontsize=xlabel_size)
            axs[2, i-4].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
        
    #fig.suptitle('Predictive Performance Comparison of Selective Rationalizations', fontsize=12)
    plt.subplots_adjust(
        left=0.09,
        bottom=0.126, 
        right=0.963, 
        top=0.983, 
        wspace=0.433, 
        hspace=0.49,
        )
    plt.legend(bbox_to_anchor=(-0.3, -0.37), loc='upper center', borderaxespad=0, fontsize=10,
                fancybox=True,ncol=3)
    #plt.xticks(fontsize=xtick_size)
    plt.show()
    fig1 = plt.gcf()
    fig.savefig('./fresh_compare_to_bert_attributes.png', dpi=600)

    exit()

        








def change_label_and_create_new_df(model_output_array, scaled_attention_path, deeplift_path, gradients_path):  #extracted_rationales_path, 
    data = arguments["dataset"]
    seed = seeds[data]
    feature = arguments["feature"]

    # model_output_array = 
    model_output_df = pd.DataFrame(np.load(model_output_array, allow_pickle= True).item()).T
    scaled_attention_df = pd.DataFrame(np.load(scaled_attention_path, allow_pickle= True).item()).T
    deeplift_df = pd.DataFrame(np.load(deeplift_path, allow_pickle= True).item()).T
    gradients_df = pd.DataFrame(np.load(gradients_path, allow_pickle= True).item()).T
    print('------ df loaded from npy example --------')
    print(deeplift_df)
    

    BERT_pred_list = []
    scaled_list = []
    deeplift_list = []
    gradients_list = []
    for i, row in enumerate(model_output_df['predicted']):
        bert_pred = row.argmax()
        scaled_pred = scaled_attention_df['predicted'][i].argmax()
        deeplift_pred = deeplift_df['predicted'][i].argmax()
        gradients_pred = gradients_df['predicted'][i].argmax()
        BERT_pred_list.append(bert_pred)
        scaled_list.append(scaled_pred)
        deeplift_list.append(deeplift_pred)
        gradients_list.append(gradients_pred)

    model_output_df['bert_pred_label'] = BERT_pred_list
    scaled_attention_df['scaled_pred_label'] = scaled_list
    deeplift_df['deeplift_pred_label'] = deeplift_list
    gradients_df['gradients_pred_label'] = gradients_list

    # print('testing scale')
    # print(scaled_attention_df.loc['test_1075', 'scaled_pred_label'])
    # print(scaled_attention_df.loc['test_1075', 'actual'])

    # print('testing bert')
    # print(model_output_df.loc['test_1075', 'bert_pred_label'])
    # print(model_output_df.loc['test_1075', 'actual'])



    fresh_bert_pred = pd.merge(model_output_df[['bert_pred_label','actual']], scaled_attention_df['scaled_pred_label'], left_index=True, right_index=True)
    fresh_bert_pred = pd.merge(fresh_bert_pred, deeplift_df['deeplift_pred_label'], left_index=True, right_index=True)
    fresh_bert_pred = pd.merge(fresh_bert_pred, gradients_df['gradients_pred_label'], left_index=True, right_index=True)

    # print('testing scale')
    # print(fresh_bert_pred.loc['test_1075', 'scaled_pred_label'])
    # print('testing bert')
    # print(fresh_bert_pred.loc['test_1075', 'bert_pred_label'])

    return fresh_bert_pred



model_output_array='./models/'+str(data)+'_full/bert-output_seed-'+str(seed)+'.npy'
scaled_attention_path='./FRESH_classifiers/'+str(data)+'_full/topk/scaled attention-bert-output_seed-scaled attention_'+str(seed)+'.npy'
deeplift_path='./FRESH_classifiers/'+str(data)+'_full/topk/deeplift-bert-output_seed-deeplift_'+str(seed)+'.npy'
gradients_attention_path='./FRESH_classifiers/'+str(data)+'_full/topk/gradients-bert-output_seed-gradients_'+str(seed)+'.npy'
full_df = change_label_and_create_new_df(model_output_array, scaled_attention_path, deeplift_path, gradients_attention_path)
full_df['Domain'] = str('Full')

model_output_array='./models/'+str(data)+'/bert-output_seed-'+str(seed)+'.npy'
scaled_attention_path='./FRESH_classifiers/'+str(data)+'/topk/scaled attention-bert-output_seed-scaled attention_'+str(seed)+'.npy'
deeplift_path='./FRESH_classifiers/'+str(data)+'/topk/deeplift-bert-output_seed-deeplift_'+str(seed)+'.npy'
gradients_attention_path='./FRESH_classifiers/'+str(data)+'/topk/gradients-bert-output_seed-gradients_'+str(seed)+'.npy'
synd_df = change_label_and_create_new_df(model_output_array, scaled_attention_path, deeplift_path, gradients_attention_path)
synd_df['Domain'] = str('SynD')

model_output_array='./models/'+str(data)+'/bert-output_seed-'+str(seed)+'-OOD-'+str(data)+'_ood1.npy'
scaled_attention_path='./FRESH_classifiers/'+str(data)+'/topk/scaled attention-bert-output_seed-scaled attention_'+str(seed)+'-OOD-'+str(data)+'_ood1.npy'
deeplift_path='./FRESH_classifiers/'+str(data)+'/topk/deeplift-bert-output_seed-deeplift_'+str(seed)+'-OOD-'+str(data)+'_ood1.npy'
gradients_attention_path='./FRESH_classifiers/'+str(data)+'/topk/gradients-bert-output_seed-gradients_'+str(seed)+'-OOD-'+str(data)+'_ood1.npy'
asyd1_df = change_label_and_create_new_df(model_output_array, scaled_attention_path, deeplift_path, gradients_attention_path)
asyd1_df['Domain'] = str('AsyD1')

model_output_array='./models/'+str(data)+'/bert-output_seed-'+str(seed)+'-OOD-'+str(data)+'_ood2.npy'
scaled_attention_path='./FRESH_classifiers/'+str(data)+'/topk/scaled attention-bert-output_seed-scaled attention_'+str(seed)+'-OOD-'+str(data)+'_ood2.npy'
deeplift_path='./FRESH_classifiers/'+str(data)+'/topk/deeplift-bert-output_seed-deeplift_'+str(seed)+'-OOD-'+str(data)+'_ood2.npy'
gradients_attention_path='./FRESH_classifiers/'+str(data)+'/topk/gradients-bert-output_seed-gradients_'+str(seed)+'-OOD-'+str(data)+'_ood2.npy'
asyd2_df = change_label_and_create_new_df(model_output_array, scaled_attention_path, deeplift_path, gradients_attention_path)
asyd2_df['Domain'] = str('AsyD2')

one_dataset = pd.concat([full_df, synd_df, asyd1_df, asyd2_df], ignore_index = True)
one_dataset.to_csv('./saved_everything/'+str(data)+'/fresh_bert_pred_compare.csv')


