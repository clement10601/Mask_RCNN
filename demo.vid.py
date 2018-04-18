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
import cv2
from tqdm import tqdm
import coco
from tools import utils
from maskrcnn import model as modellib
from maskrcnn import visualize
import colorsys
import random
import glob

###
### Para Setup
###
vid_file = 'test.mp4'
#vid_out = 'test_out.mp4'
in_dir = './test_video'
out_dir = './test_output_video'

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_DIR = os.path.join(ROOT_DIR, "weights")
if not os.path.exists(COCO_MODEL_DIR):
    os.makedirs(COCO_MODEL_DIR)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
#config.display()

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
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
	
start_t = timeit.default_timer()

# load each video name
for roots, dirs, files in os.walk(in_dir, topdown=False):
    for vid_file in files:

        #file_names = next(os.walk(IMAGE_DIR))[2]
        
        #file_name = vid_file
        file_name = os.path.join(roots, vid_file)
        print('Processing file:{}'.format(file_name))
        
        video_reader = cv2.VideoCapture(file_name)
        fps = video_reader.get(cv2.CAP_PROP_FPS)
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        out_file_name = vid_file.split('.mp4')[0] + '_0.8_out.mp4'
        vid_out = os.path.join(out_dir, out_file_name)
        print(vid_out)
        
        video_writer = cv2.VideoWriter(vid_out,
                                       cv2.VideoWriter_fourcc(*'MP4V'), 
                                       fps, 
                                       (frame_w, frame_h))
        print("Processing {0} frames, FPS: {1}".format(nb_frames, fps))
        colors = random_colors(100)
        for i in tqdm(range(nb_frames)):
            ret, image = video_reader.read()
            results = model.detect([image], verbose=0)
            r = results[0]
            image = visualize.return_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'], score_throttle='0.8', colors=colors)
            video_writer.write(np.uint8(image))
        video_reader.release()
        video_writer.release()
        
stop_t = timeit.default_timer()
print("Exec Time: {}".format(stop_t - start_t))
