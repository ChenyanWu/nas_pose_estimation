#!/bin/bash
mkdir data
cd data
mkdir coco
cd coco
mkdir person_detection_results
mkdir annotations
mkdir images
#wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip -d images/
unzip annotations_trainval2017.zip -d annotations/