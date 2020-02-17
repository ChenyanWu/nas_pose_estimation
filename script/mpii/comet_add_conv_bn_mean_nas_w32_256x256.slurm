#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node=24
#SBATCH --output=/home/chenyan/coding/nas_pose_estimation/script/log_script/mpii/add_conv_bn_mean_nas_w32_256x192.out
#SBATCH -t 36:00:00
module purge
module load cuda
date
source /home/chenyan/anaconda3/etc/profile.d/conda.sh
conda activate py37

set -x
cd /home/chenyan/coding/nas_pose_estimation
time python tools/train.py --cfg experiments/mpii/nas_hrnet/add_conv_bn_mean_w32_256x256_adam_lr1e-3.yaml

date