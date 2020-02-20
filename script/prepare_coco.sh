#!/bin/bash
cd /scratch/chenyan
cd $(ls -d */|head -n 1)
cp -r /oasis/scratch/comet/chenyan/temp_project/coco ./
cd coco/images
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
echo done coco dataset