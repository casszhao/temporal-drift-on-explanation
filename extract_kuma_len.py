import numpy as np
import logging
import os

dataset = 'xfact'

log_dir = "saved_everything/" + str(dataset)
os.makedirs(log_dir, exist_ok = True)
logging.basicConfig(
    filename= log_dir + "/kuma_length.log", 
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info(f'''
        {dataset} ----''')

def one_domain_len(domain):
    overall = np.zeros(5)
    for _j_, seed in enumerate([5,10,15,20,25]):
        if 'ood' in str(domain):
            path_to_file : str = f'kuma_model/{dataset}/kuma-bert-output_seed-kuma-bert{seed}-OOD-{dataset}_{domain}.npy'
        elif 'full' in str(domain):
            path_to_file : str = f'kuma_model/{dataset}_full/kuma-bert-output_seed-kuma-bert{seed}.npy'
        else:
            path_to_file : str = f'kuma_model/{dataset}/kuma-bert-output_seed-kuma-bert{seed}.npy'
        
        file_data = np.load(path_to_file, allow_pickle=True).item()
        aggregated_ratio = np.zeros(len(file_data))

        for _i_, (docid, metadata) in enumerate(file_data.items()):

            rationale_ratio = min( 
                1.,
                metadata['rationale'].sum()/metadata['full text length']
            )

            aggregated_ratio[_i_] = rationale_ratio

        overall[_j_] = aggregated_ratio.mean()
    
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

one_domain_len('full')
one_domain_len('InDomain')
one_domain_len('ood1')
one_domain_len('ood2')