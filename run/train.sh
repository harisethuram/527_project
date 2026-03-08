#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu
#SBATCH --output=/gscratch/ark/hari/527_project/s_out/h_param_search_%j.out

source activate chexpert

python train.py \
    --model $1 \
    --batch_size $2 \
    --lr $3 \
    --epochs 1 \
    --output_dir /gscratch/ark/hari/527_project/models/hparam/${1}/${2}/${3}/