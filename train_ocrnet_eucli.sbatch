#!/bin/bash
#SBATCH --job-name=""
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:20G
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=
#SBATCH --error=
srun tools/dist_train.sh configs/ocrnet/ocrnet_hr48_4xb2-80k_cityscapes-512x1024_euclidean.py 4
