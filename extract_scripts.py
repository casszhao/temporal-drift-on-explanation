import numpy as np

dataset = 'complain'

overall = np.zeros(5)

for _j_, seed in enumerate([5,10,15,20,25]):

    path_to_file : str = f'kuma_model/{dataset}/kuma-bert-output_seed-kuma-bert{seed}-OOD-{dataset}_ood1.npy'

    file_data = np.load(path_to_file, allow_pickle=True).item()

    aggregated_ratio = np.zeros(len(file_data))

    for _i_, (docid, metadata) in enumerate(file_data.items()):

        rationale_ratio = min( 
            1.,
            metadata['rationale'].sum()/metadata['full text length']
        )

        aggregated_ratio[_i_] = rationale_ratio

    overall[_j_] = aggregated_ratio.mean()

print(f'''
    mean -> {overall.mean()}
    std ->  {overall.std()}
    all ->  {overall}
''')