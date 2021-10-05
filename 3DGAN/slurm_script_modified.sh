#! /bin/bash
#SBATCH --job-name=3DGAN
#SBATCH -n 40
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

module load nvidia/cuda/11.0
. /apps/local/conda_init.sh 
conda activate 3dgan

nvidia-smi
hostname

python train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0,1,2,3 --dataroot=./data/LIDC-HDF5-256 --dataset=train --tag=d2_multiview2500 --data=LIDC256 --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/train.txt --valid_datasetfile=./data/test.txt --valid_dataset=test
