#!/bin/bash
#SBATCH --job-name="farid-run"
#SBATCH --output="/home/alham.fikri/farid/outputs/run_semeval-lingualchemy-masakhanews.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -p A100                     # Use the gpu partition
#SBATCH --reservation=gpu-a100-abdul
#SBATCH --gres=gpu:4



nvidia-smi
hostname
# CUDA_VISIBLE_DEVICES=6,7 python masakhanews_baseline.py --exp_name mbert-masakhanews-baseline --model bert-base-multilingual-cased --output-dir /mnt/beegfs/farid/lingualchemy/masakhanews/mbert-base
# CUDA_VISIBLE_DEVICES=6,7 python masakhanews_baseline.py --exp_name xlmr-base-masakhanews-baseline --model xlm-roberta-base --output-dir /mnt/beegfs/farid/lingualchemy/masakhanews/xlmr-base
# CUDA_VISIBLE_DEVICES=6,7 python masakhanews_baseline.py --exp_name xlmr-large-masakhanews-baseline --model xlm-roberta-large --output-dir /mnt/beegfs/farid/lingualchemy/masakhanews/xlmr-large


# CUDA_VISIBLE_DEVICES=6,7 python masakhanews_lingualchemy.py --exp_name mbert-masakhanews-lingualchemy --model bert-base-multilingual-cased --output-dir /mnt/beegfs/farid/lingualchemy/masakhanews/mbert-base-lingualchemy
# CUDA_VISIBLE_DEVICES=6,7 python masakhanews_lingualchemy.py --exp_name xlmr-base-masakhanews-lingualchemy --model xlm-roberta-base --output-dir /mnt/beegfs/farid/lingualchemy/masakhanews/xlmr-base-lingualchemy
CUDA_VISIBLE_DEVICES=6,7 python masakhanews_lingualchemy.py --exp_name xlmr-large-masakhanews-lingualchemy --model xlm-roberta-large --output-dir /mnt/beegfs/farid/lingualchemy/masakhanews/xlmr-large-lingualchemy