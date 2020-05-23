#!/bin/bash
#make model dir
mkdir -p models/pytorch/imagenet
mkdir -p models/pytorch/imagenet_covariance_hrnet

#install python package
pip install -r requirements.txt
echo done python package install

#install coco api
cd ..
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make install
cd ../../nas_pose_estimation
echo done coco api install

#make lib
cd lib
make
cd ..

#download dataset
mkdir data && cd data
mkdir coco && cd coco
mkdir person_detection_results
mkdir images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip -d images/
unzip train2017.zip -d images/
unzip annotations_trainval2017.zip
cd ../..
echo done dataset install