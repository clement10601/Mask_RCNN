#!/bin/bash

# note you must use git clone --recursive to clone source code from https://192.168.11.111/ObjectDetection/Mask_RCNN.git

# setup a virtual environment
virtualenv -p python3 mask_rcnn_virtualenv

source mask_mask_rcnn_virtualenv/bin/activate

# install requirement package
pip install -r requirements.txt

# clone cocoapi
git clone https://192.168.11.111/ObjectDetection/cocoapi.git

# make cocoapi
cd cocoapi
make

