#!/bin/bash
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00

# set name of job
#SBATCH --job-name=yelp_BuLSTM

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
source activate ood_faith

dataset="yelp"
model_dir="models/"
data_dir="datasets/"
evaluation_dir="posthoc_results/"
extracted_rationale_dir="extracted_rationales/"
thresholder="topk"


# # ###### Train BERT #########  on full dataset and In Domain
# # ##########################################################
# for seed in 5 10 15 20 25
# do
#    python train_fulltext_and_kuma.py --dataset $dataset$"_full" --model_dir $model_dir --data_dir $data_dir --seed $seed
#    python train_fulltext_and_kuma.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --seed $seed
# done
# echo "done TRAINING BERT on full data and In Domain"
# python train_fulltext_and_kuma.py --dataset $dataset$"_full" --model_dir $model_dir --data_dir $data_dir --evaluate_models
# python train_fulltext_and_kuma.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluate_models
# echo "done EVALUATION BERT on full data and In Domain"


# ##### evaluate POSTHOC BERT for full data and in domain
# python evaluate_posthoc.py --dataset $dataset$"_full" --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder topk
# python evaluate_posthoc.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder topk
# echo "done evaluate faithfulness for topk for both full and indmain"
# python evaluate_posthoc.py --dataset $dataset$"_full" --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder contigious
# python evaluate_posthoc.py --dataset $dataset --model_dir $model_dir --data_dir $data_dir --evaluation_dir $evaluation_dir --thresholder contigious
# echo "done evaluate faithfulness for contigious for both full and indmain"


######## Train Fresh ################### on full dataset and In Domain
######################################################################

##### extract rationales
# python FRESH_extract_rationales_cass.py --dataset $dataset$"_full" --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder
# python FRESH_extract_rationales_cass.py --dataset $dataset$"_full" --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder contigious
# python FRESH_extract_rationales_cass.py --dataset $dataset --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder
# python FRESH_extract_rationales_cass.py --dataset $dataset --data_dir $data_dir --model_dir $model_dir --extracted_rationale_dir $extracted_rationale_dir --thresholder contigious
# echo 'done extract rationales (top and contigious) for FRESH'



# ##### train FRESH for top and contigious
# for importance_metric in  "attention" "gradients" "lime" "deeplift"
# do
#   echo 'starting training FRESH with: '
#   echo $importance_metric
#   echo $thresholder
#       for seed in 5 10 15 20 25
#       do
#           python FRESH_train_on_rationales.py --dataset $dataset$"_full" --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric --seed $seed
#           python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric --seed $seed
#       done
#       echo 'Done training FRESH for:'
#       echo $importance_metric
#       echo $thresholder
#       python FRESH_train_on_rationales.py --dataset $dataset$"_full" --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
#       python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
#       echo 'Done evaluating FRESH for:'
#       echo $importance_metric
#       echo $thresholder
# done
# ### scaled attention
# echo "starting training FRESH with: scaled attention"
# for seed in 5 10 15 20 25
# do
#     python FRESH_train_on_rationales.py --dataset $dataset$"_full" --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --seed $seed
#     python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --seed $seed
# done
# echo "starting evaluating FRESH with: scaled attention"
# python FRESH_train_on_rationales.py --dataset $dataset$"_full" --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --evaluate_models
# python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --evaluate_models

# thresholder="contigious"
# for importance_metric in  "attention" "gradients" "lime" "deeplift"
# do
#   echo 'starting training FRESH with: '
#   echo $importance_metric
#   echo $thresholder
#       for seed in 5 10 15 20 25
#       do
#           python FRESH_train_on_rationales.py --dataset $dataset$"_full" --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric --seed $seed
#           python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric --seed $seed
#       done
#       echo 'Done training FRESH for:'
#       echo $importance_metric
#       echo $thresholder
#       python FRESH_train_on_rationales.py --dataset $dataset$"_full" --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
#       python FRESH_train_on_rationales.py --dataset $dataset --rationale_model_dir "FRESH_classifiers/" --extracted_rationale_dir $extracted_rationale_dir --thresholder $thresholder --importance_metric $importance_metric  --evaluate_models
#       echo 'Done evaluating FRESH for:'
#       echo $importance_metric
#       echo $thresholder
# done
# #### scaled attention
# echo "starting training FRESH with: scaled attention"
# for seed in 5 10 15 20 25
# do
#     python FRESH_train_on_rationales.py --dataset $dataset$"_full" --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --seed $seed
#     python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --seed $seed
# done
# echo "starting evaluating FRESH with: scaled attention"
# python FRESH_train_on_rationales.py --dataset $dataset$"_full" --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --evaluate_models
# python FRESH_train_on_rationales.py --dataset $dataset --extracted_rationale_dir $extracted_rationale_dir --rationale_model_dir "FRESH_classifiers/" --thresholder $thresholder --importance_metric "scaled attention" --evaluate_models







########################## Train LSTM

########## train and test on full dataset
echo "start TRAINING LSTM on full data"
for seed in 5 10 15 20 25
do
   python train_fulltext_and_kuma.py --dataset $dataset$"_full" --model_dir LSTM_model --data_dir $data_dir --seed $seed --inherently_faithful "full_lstm"
   python train_fulltext_and_kuma.py --dataset $dataset --model_dir LSTM_model --data_dir $data_dir --seed $seed --inherently_faithful "full_lstm"
done
echo "done TRAINING LSTM on full data and indomain data"

python train_fulltext_and_kuma.py --dataset $dataset$"_full" --model_dir LSTM_model --data_dir $data_dir --evaluate_models --inherently_faithful "full_lstm"
python train_fulltext_and_kuma.py --dataset $dataset --model_dir LSTM_model --data_dir $data_dir --evaluate_models --inherently_faithful "full_lstm"
echo "done EVALUATION LSTM on full data and indomain data"




############ train KUMA on FULL DATASET ######

# echo '-------- start training kuma on full data------------'
# for seed in 5 10 15 20 25
# do
# python train_fulltext_and_kuma.py --dataset $dataset$"_full" --model_dir "kuma_model/" --data_dir $data_dir --seed $seed --inherently_faithful "kuma"
# done
# echo "done train kuma on full data"
# python train_fulltext_and_kuma.py --dataset $dataset$"_full" --model_dir "kuma_model/" --data_dir $data_dir --inherently_faithful "kuma" --evaluate_models
# echo "done eval kuma on full data"

# echo '-------- start training kuma on in domain------------'
# for seed in 5 10 15 20 25
# do
#    python train_fulltext_and_kuma.py --dataset $dataset --model_dir "kuma_model/" --data_dir $data_dir --seed $seed --inherently_faithful "kuma"
# done
# echo "done train kuma"
# python train_fulltext_and_kuma.py --dataset $dataset --model_dir "kuma_model/" --data_dir $data_dir --seed $seed --inherently_faithful "kuma" --evaluate_models
# echo "done eval kuma"

# python extract_kuma_len.py --dataset $dataset
# echo "done extract kuma len"





##### change name for
# cd ./FRESH_classifiers/
# echo 'go change name for scaled attention'
# shopt -s globstar
# for file in **/*\ *
# do
#     mv "$file" "${file// /_}"
# done
# cd ../../


# ##### scaled\ attention
# python save_predictive.py --dataset $dataset
# python save_everything.py --dataset $dataset --save_for_kuma_lstm
# python save_similarity.py --dataset $dataset





