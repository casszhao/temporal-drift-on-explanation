#!/bin/bash

#module load python/anaconda3
#module load cuda/11.4
#source activate pythonProject

for seed in 5 10 15 20 25
do
python train_bert_only_full_data.py --dataset xfact_full --data_dir datasets/ --model_dir models/ --seed $seed
done
python train_bert_only_full_data.py --dataset xfact_full --data_dir datasets/ --model_dir models/ --evaluate_models
