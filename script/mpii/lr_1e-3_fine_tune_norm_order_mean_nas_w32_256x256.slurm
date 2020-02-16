#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-AI
#SBATCH --gres=gpu:volta16:4
#SBATCH --output=/pylon5/ir5fp5p/chenyan/coding/eccv2020/nas_pose_estimation/script/log_script/mpii/lr1e-3_fine_tune_norm_order_nas_w32_256x192.out
#SBATCH -t 36:00:00
date
bash
source /pylon5/ir5fp5p/chenyan/opt/anaconda3/etc/profile.d/conda.sh
conda activate open-mmlab
set -x
cd /pylon5/ir5fp5p/chenyan/coding/eccv2020/nas_pose_estimation
time python tools/train.py --cfg experiments/mpii/nas_hrnet/lr_1e-3_fine_tune_norm_order_w32_256x256_adam_lr1e-3.yaml

date