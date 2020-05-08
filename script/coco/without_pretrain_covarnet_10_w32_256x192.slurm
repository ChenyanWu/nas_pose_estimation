#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-AI
#SBATCH --gres=gpu:volta16:4
#SBATCH --output=/pylon5/ir5fp5p/chenyan/coding/eccv2020/nas_pose_estimation/script/log_script/coco/pretrain_covariance_10_hrnet_w32_256x192_adam_lr1e-3.out
#SBATCH -t 47:00:00
date
source /pylon5/ir5fp5p/chenyan/opt/anaconda3/etc/profile.d/conda.sh
conda activate open-mmlab

set -x
cd /pylon5/ir5fp5p/chenyan/coding/eccv2020/nas_pose_estimation
time python tools/train.py --cfg experiments/coco/covariance_hrnet/number10_without_pretrain_w32_256x192_adam_lr1e-3.yaml
date