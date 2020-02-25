#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --output=/home/chenyan/coding/nas_pose_estimation/script/log_script/coco/double_resume_mean_w48_384x288_adam_lr1e-3.out
#SBATCH -t 47:00:00
module purge
module load cuda
date
source /home/chenyan/anaconda3/etc/profile.d/conda.sh
conda activate py37

set -x
cd /scratch/chenyan
cd $(ls -d */|head -n 1)
cp -r /oasis/scratch/comet/chenyan/temp_project/coco ./
cd coco/images
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
echo done coco dataset
cd ../..
coco_path=$(pwd)/coco/
cd /home/chenyan/coding/nas_pose_estimation
time python tools/train.py --cfg experiments/coco/nas_hrnet/mean_w48_384x288_adam_lr1e-3.yaml DATASET.ROOT $coco_path

date
