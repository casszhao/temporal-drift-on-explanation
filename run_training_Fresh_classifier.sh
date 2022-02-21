#!/bin/bash

#module load python/anaconda3
#module load cuda/11.4
#source activate pythonProject

for seed in 10 15 20 25
do
python train_fulltext_and_kuma.py --dataset xfact --data_dir datasets/ --model_dir models/ --seed $seed
done
python train_fulltext_and_kuma.py --dataset xfact --data_dir datasets/ --model_dir models/ --evaluate_models
