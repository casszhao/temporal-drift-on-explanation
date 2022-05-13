#!/bin/bash
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=6-00:00

# set name of job
#SBATCH --job-name=FTBert-healthfact

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


dataset="healthfact"
model_dir="models/"
data_dir="datasets/"
evaluation_dir="posthoc_results/"
extracted_rationale_dir="extracted_rationales/"
thresholder="topk"

python python ft_hp_for_bert.py --dataset $dataset