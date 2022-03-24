#!/bin/bash

module load python/anaconda3
module load cuda/10.2
source activate ood_faith

dataset="complain"
model_dir="models/"
data_dir="datasets/"
evaluation_dir="evaluating_faithfulness/"
extracted_rationale_dir="extracted_rationales/"
rationale_model_dir="rationale_models/"
thresholder="topk"



for seed in 5 10 15 20 25
do
   python train_fulltext_and_kuma.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --seed $seed
done
echo "done TRAINING bert on full text"
python train_fulltext_and_kuma.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluate_models
echo "done EVALUATION bert on full text"

for seed in 5 10 15 20 25
do
python train_fulltext_and_kuma.py --dataset $dataset --model_dir "kuma_model_new" --data_dir $data_dir --seed $seed --inherently_faithful "kuma"
done
echo "done train kuma"
python train_fulltext_and_kuma.py --dataset $dataset --model_dir "kuma_model_new" --data_dir $data_dir --seed $seed --inherently_faithful "kuma" --evaluate_models
echo "done eval kuma"

python evaluate_posthoc.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder topk
python evaluate_posthoc.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder contigious
echo "done evaluate faithfulness"

python FRESH_extract_rationales_no_ood.py --dataset $dataset --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder
python FRESH_extract_rationales_no_ood.py --dataset $dataset --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder contigious
echo 'done extract rationales for FRESH'

############################### DONT NEED THIS PART #######################
# python extract_rationales.py --dataset $dataset  --model_dir $model_dir --data_dir $data_dir --extracted_rationale_dir $extracted_rationale_dir --use_tasc
# python extract_rationales.py --dataset $dataset  --model_dir $model_dir --data_dir $data_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder contigious --use_tasc
###############################




for importance_metric in  "attention" "ig" "gradients" "lime" "deeplift"
do
  echo 'starting training FRESH with: '
  echo $importance_metric
  echo $thresholder
      for seed in 5 10 15 20 25
      do
          python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric $importance_metric
      done
      python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
done
##
echo "scaled attention"
for seed in 5 10 15 20 25
do
    python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric "scaled attention"
done
python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric "scaled attention"  --evaluate_models



thresholder="contigious"


for importance_metric in  "attention" "ig" "gradients" "lime" "deeplift"
do
  echo 'starting training FRESH with: '
  echo $importance_metric
  echo $thresholder
      for seed in 5 10 15 20 25
      do
          python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric $importance_metric
      done
      python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
done
#
echo "scaled attention"
for seed in 5 10 15 20 25
do
    python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric "scaled attention"
done
python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --seed $seed --thresholder $thresholder --importance_metric "scaled attention"  --evaluate_models

