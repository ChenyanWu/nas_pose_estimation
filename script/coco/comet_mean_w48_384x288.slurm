#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --output=/home/chenyan/coding/nas_pose_estimation/script/log_script/coco/mean_w48_384x288_adam_lr1e-3.out
#SBATCH -t 47:00:00
module purge
module load cuda
date
source /home/chenyan/anaconda3/etc/profile.d/conda.sh
conda activate py37

set -x
cd /home/chenyan/coding/nas_pose_estimation
time python tools/train.py --cfg experiments/coco/nas_hrnet/mean_w48_384x288_adam_lr1e-3.yaml

date
