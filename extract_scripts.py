import numpy as np

path_to_file : str = 'kuma_model/complain/kuma-bert-output_seed-kuma-bert5-OOD-complain_ood1.npy'

file_data = np.load(path_to_file, allow_pickle=True).item()

aggregated_ratio = np.zeros(len(file_data))

for _i_, (docid, metadata) in enumerate(file_data.items()):

    rationale_ratio = min( 
        1.,
        metadata['rationale'].sum()/metadata['full text length']
    )

    aggregated_ratio[_i_] = rationale_ratio


import pdb; pdb.set_trace()