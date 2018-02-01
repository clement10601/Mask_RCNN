import timeit
import os
import sys
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import skimage.io
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()

## Load COCO dataset
#dataset = coco.CocoDataset()
#dataset.load_coco("data", "train")
#dataset.prepare()

# Print class names
#class_names = dataset.class_names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

file_names = next(os.walk(IMAGE_DIR))[2]
#file_names = ['gggggg.jpg']
start_t = timeit.default_timer()
sum_dect = 0.0
for idx, name in enumerate(file_names):
    f_name = os.path.join(IMAGE_DIR, name)
    image = skimage.io.imread(f_name)

    # Run detectionV
    local_t_str = timeit.default_timer()
    results = model.detect([image], verbose=1)
    local_t_end = timeit.default_timer()
    d_time = local_t_end - local_t_str
    sum_dect += d_time
    print("Dect Time: {}".format(d_time))

    # Visualize results
    r = results[0]
    f = os.path.basename(f_name)
    visualize.save_instances(f, image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
stop_t = timeit.default_timer()
print("Avg. Dect Time: {}".format(sum_dect / len(file_names)))
print("Exec Time: {}".format(stop_t - start_t))
