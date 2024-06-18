#!/bin/bash
#SBATCH --job-name=""
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=
#SBATCH --error=


srun tools/dist_test.sh configs/deeplabv3plus/ood_mapillary_eucli.py pretrained/deeplab_eucli/iter_80000.pth  1 --eval mIoU --T 1.0 --numclasses 7 --numbins 20 --infermode softmax --ecemode cwece










