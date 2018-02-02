import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import skimage.io
import matplotlib.pyplot as plt

from tools import utils
from tools.config import Config
from maskrcnn import model as modellib
from maskrcnn import visualize

parser = argparse.ArgumentParser()
parser.add_argument("path", help="input path")
args = parser.parse_args()
if args.path:
  try:
    INPUT_PATH = os.path.abspath(args.path)
  except Exception as e:
    print(e)
    sys.exit(1)
else:
  raise Exception('No input file!')
  sys.exit(1)
###
### Para Setup
###
# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
config = InferenceConfig()

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

path, filename = os.path.split(INPUT_PATH)
image = skimage.io.imread(INPUT_PATH)
# Run detectionV
results = model.detect([image], verbose=0)

# Visualize results
r = results[0]

labels = []
labels = visualize.extract_instances(r['rois'], r['masks'], r['class_ids'],
                        class_names, r['scores'], score_throttle='0.9')
stdout = "{{\"code\":200, \"msg\":\"Upload success.\",\"imgurl\":\"{}\",\"result\":\"{}\"}}"
print(stdout.format(filename, labels))

