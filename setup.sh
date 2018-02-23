#!/bin/bash

# note you must use git clone --recursive to clone source code from https://192.168.11.111/ObjectDetection/Mask_RCNN.git

# set sslverify to false
git config --global http.sslverify false

apt-get install -y python-setuptools
apt install -y python-pip
apt-get install -y python3-dev

# setup a virtual environment
virtualenv -p python3 mask_rcnn_virtualenv

source mask_rcnn_virtualenv/bin/activate

# install requirement package
/home/john/Mask_RCNN/mask_rcnn_virtualenv/bin/pip install -r requirements.txt

# clone cocoapi
git clone https://192.168.11.111/ObjectDetection/cocoapi.git

cd cocoapi
/home/john/Mask_RCNN/mask_rcnn_virtualenv/bin/pip install -r requirements.txt

# make cocoapi
cd PythonAPI
echo '------------ make cocoapi--------------'
pwd
make

