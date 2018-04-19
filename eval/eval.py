import os
import sys
import csv
import numpy as np
import time
import argparse

sys.path.append('/home/john/object_detect/Mask_RCNN/')
sys.path.append('/home/john/object_detect/Mask_RCNN/maskrcnn/')
sys.path.append('/home/john/object_detect/Mask_RCNN/cocoapi/PythonAPI/')

import matplotlib
matplotlib.use('Agg')
#%matplotlib inline 
import matplotlib.pyplot as plt

import utils
import skimage.io
from maskrcnn import model as modellib
from maskrcnn import visualize
from model import log

# MS COCO Dataset
import coco
print('coco lib path:{}'.format(coco.__file__))

# Coco Dataset Config
config = coco.CocoConfig()

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# parse args
parser = argparse.ArgumentParser(
        description="eval coco object detection performance")

parser.add_argument("--model_name", required=False,
        default="mask_rcnn_coco.h5",
        help="coco model name")

parser.add_argument("--result_name", required=True,
        default="coco_valid_result.csv",
        help="valid result file name")

parser.add_argument("--score_threshold", required=False,
	default=0.95,
	help="score threshold")


args = parser.parse_args()

SCORE_THRESHOLD = args.score_threshold

# define path of coco dataset
COCO_DIR = "/workspace/coco/2017"

### Setup Model
# Root directory of the project
ROOT_DIR = "/home/john/object_detect/Mask_RCNN"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_DIR = os.path.join(ROOT_DIR, "weights")
if not os.path.exists(COCO_MODEL_DIR):
        os.makedirs(COCO_MODEL_DIR)
COCO_FILE = os.path.join("weights/", args.model_name)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, COCO_FILE)
print(COCO_MODEL_PATH)

#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco_0159.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
print("!!! Loading weights of model !!!")
model.load_weights(COCO_MODEL_PATH, by_name=True)


### Load Dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    print('start load coco model {}'.format(COCO_DIR))
    dataset.load_coco(COCO_DIR, "val")


# Prepare data
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))

"""
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
"""

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def get_unique_id_list(input_list):
    """
    get unique id in input list,
    filter duplicate item, return >=0 interger
    """
    rtn_list = []

    for i in input_list:
        if i not in rtn_list and i >= 0:
            rtn_list.append(i)
    return rtn_list

# Load and display random samples
print("dataset.image_id len=>", len(dataset.image_ids))

#image_ids = np.random.choice(dataset.image_ids, 10)

TP_total = np.zeros(dataset.num_classes)
FP_total = np.zeros(dataset.num_classes)
FN_total = np.zeros(dataset.num_classes)

AP_total = []

# used to calculate precision of detection
Detection_True_cnt = 0
Detection_total_cnt = 0

start_time = time.time()

#image_ids = np.random.choice(dataset.image_ids, 4)
#for image_id in image_ids:

for image_id in dataset.image_ids:
    print('----------------------------------')
    image = dataset.load_image(image_id)
   
    # Parse image additional info           
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("Processing image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       dataset.image_reference(image_id)))
    
    # Detect
    results = model.detect([image], verbose=0)
    ax = get_ax(1)
    r = results[0]

    # Calculate each object accuracy
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, 
                                          r['rois'], r['class_ids'], r['scores'])
    print("AP =====> ", AP)
    #print("precisions", precisions)
    #print("recalls", recalls)
    AP_total.append(AP)

    
    TP, FP, FN , Detect_true_cnt = utils.compute_class_PR(gt_bbox, gt_class_id,
                                        r['rois'], r['class_ids'], r['scores'],
                                        dataset.num_classes, 0.51, 0.95)

    #print("gt_class_id", gt_class_id)
    #print("pred_class_id", r['class_ids'])
    #print("TP list", TP)
    #print("FP list ", FP)
    #print("FN list", FN)
    TP_total = np.add(TP_total, TP)
    FP_total = np.add(FP_total, FP)
    FN_total = np.add(FN_total, FN)

    Detection_True_cnt += Detect_true_cnt
    Detection_total_cnt += len(gt_bbox)

# Calculate detection precision
Detection_Precision = (float(Detection_True_cnt) / Detection_total_cnt)

# Calculate mean Precision and Recall of each class
print("TP_total=>", TP_total)
print("FP_total=>", FP_total)
print("FN_total=>", FN_total)

TP_add_FP = np.add(TP_total, FP_total)
TP_add_FN = np.add(TP_total, FN_total)

Precision_AVG = np.divide(TP_total, TP_add_FP, out=np.zeros_like(TP_total), where=TP_add_FP!=0)
print("Precision of each classes =>", Precision_AVG)

Recall_AVG = np.divide(TP_total, TP_add_FN, out=np.zeros_like(TP_total), where=TP_add_FN!=0)
print("Recall of each classes =>", Recall_AVG)

print("Mean of each class mAP = >", np.mean(AP_total))

print("Detection Precision =>", Detection_Precision)

with open(args.result_name, 'w') as f:
    cursor = csv.writer(f)
    csv_header = ['class name', 'precision', 'recall']
    cursor.writerow(csv_header)

    for idx, info in enumerate(dataset.class_info):
        row_info = [info['name'], Precision_AVG[idx], Recall_AVG[idx]]
        cursor.writerow(row_info)
    
    cursor.writerow(['Detection_Precision', Detection_Precision, 'mean_of_class_mAP', np.mean(AP_total)])


end_time = time.time()
print("Cost time =>", (end_time - start_time), "s")
