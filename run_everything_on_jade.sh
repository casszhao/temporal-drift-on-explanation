#!/bin/bash
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00

# set name of job
#SBATCH --job-name=complain

# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --partition=small

#SBATCH --mem=60GB

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=zhixue.zhao@sheffield.ac.uk

# run the application
cd /jmain02/home/J2AD003/txk58/zxz22-txk58/extract_rationales/extract_rationales/
module load python/anaconda3
module load cuda/10.2
source activate time_ood

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


echo "start training kuma"
for seed in 5 10 15 20 25
do
python train_fulltext_and_kuma.py --dataset $dataset --model_dir "kuma_model_new" --data_dir $data_dir --seed $seed --inherently_faithful "kuma"
done
echo "done train kuma"
python train_fulltext_and_kuma.py --dataset $dataset --model_dir "kuma_model_new" --data_dir $data_dir --inherently_faithful "kuma" --evaluate_models
echo "done eval kuma"


python evaluate_posthoc.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder topk
python evaluate_posthoc.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder contigious
echo "done evaluate faithfulness"

python FRESH_extract_rationales_no_ood.py --dataset $dataset --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder
python FRESH_extract_rationales_no_ood.py --dataset $dataset --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder contigious
echo 'done extract rationales for FRESH'


for importance_metric in  "attention" "ig" "gradients" "lime" "deeplift"
do
  echo 'starting training FRESH with: '
  echo $importance_metric
  echo $thresholder
      for seed in 5 10 15 20 25
      do
          python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric --seed $seed
      done
      echo 'evaluate FRESH for:'
      echo $importance_metric
      echo $thresholder
      python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
done
############### scaled attention
echo "starting training FRESH with: scaled attention"
for seed in 5 10 15 20 25
do
    python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir $rationale_model_dir --thresholder $thresholder --importance_metric "scaled attention" --seed $seed
done
echo "starting evaluating FRESH with: scaled attention"
python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir $rationale_model_dir --thresholder $thresholder --importance_metric "scaled attention" --evaluate_models

#


thresholder="contigious"
for importance_metric in  "attention" "ig" "gradients" "lime" "deeplift"
do
  echo 'starting training FRESH with: '
  echo $importance_metric
  echo $thresholder
      for seed in 5 10 15 20 25
      do
          python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric --seed $seed
      done
      echo 'evaluate FRESH for:'
      echo $importance_metric
      echo $thresholder
      python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir $rationale_model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
done
############### scaled attention
echo "starting training FRESH with: scaled attention"
for seed in 5 10 15 20 25
do
    python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir $rationale_model_dir --thresholder $thresholder --importance_metric "scaled attention" --seed $seed
done
echo "starting evaluating FRESH with: scaled attention"
python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir $rationale_model_dir --thresholder $thresholder --importance_metric "scaled attention" --evaluate_models
