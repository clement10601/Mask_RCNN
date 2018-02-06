# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse
import atexit
import getpass
import ssl
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('PATH', '')
import skimage.io

from .tools import utils
from .tools.config import Config
from .maskrcnn import model as modellib
from .maskrcnn import stdinstance


class MLService(object):
    def __init__(self, app=None):
        if app is not None:
            self.app = app
            self.init_app(self.app)
        else:
            self.app = None
        # Root directory of the project
        self.ROOT_DIR = app.config['ROOT_DIR']
        # Directory to save logs and trained model
        self.MODEL_DIR = app.config['MODEL_DIR']
        # Local path to trained weights file
        self.COCO_MODEL_DIR = app.config['COCO_MODEL_DIR']
        self.COCO_MODEL_PATH = app.config['COCO_MODEL_PATH']
        self.config = InferenceConfig()
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                            'bus', 'train', 'truck', 'boat', 'traffic light',
                            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                            'kite', 'baseball bat', 'baseball glove', 'skateboard',
                            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                            'teddy bear', 'hair drier', 'toothbrush']
        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)
        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)


    def init_app(self, app):
        self.app = app

    def detect(self, IMG_PATH, throttle='0.9'):
      INPUT_PATH = os.path.abspath(IMG_PATH)
      path, filename = os.path.split(INPUT_PATH)
      image = skimage.io.imread(INPUT_PATH)
      # Run detectionV
      results = self.model.detect([image], verbose=0)
      # Visualize results
      r = results[0]
      labels = []
      labels = stdinstance.extract_instances(r['rois'], r['masks'], r['class_ids'],
                                             self.class_names, r['scores'], score_throttle=throttle)
      return {"imgurl": filename, "result": labels}


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

