#!/bin/bash
#SBATCH --job-name=""
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:40G
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=
#SBATCH --error=
srun tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_euclidean.py 4
