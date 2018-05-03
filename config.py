import os
import urllib.request
import shutil
from mrcnn.config import Config

VERBOSE = True

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

ROOT_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "images")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
# Local path to trained weights file
COCO_MODEL_DIR = os.path.join(ROOT_DIR, "weights")
if not os.path.exists(COCO_MODEL_DIR):
    os.makedirs(COCO_MODEL_DIR)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    download_trained_weights(COCO_MODEL_PATH, verbose=VERBOSE)

import GPUtil

LEAST_GMEM = 2250  # MB
MAX_THREADS = 1
MIN_FRAC = 0.3
MAX_FRAC = 0.3
GPU_LAOD = 0.5
GMEM_LAOD_LIMIT = 1.0
AVAIL_DEVICE_LIST = []
AVAIL_DEVICE_MAT = []
AVAIL_DEVICE_MEMFRAC = []
AVAIL_DEVICE_MAXTHREAD = []
try:
    GPUs = GPUtil.getGPUs()
    Gall = ''
    Gfree = ''
    for GPU in GPUs:
        Gall = GPU.memoryTotal
        Gfree = GPU.memoryFree
        GMEM_LAOD_LIMIT = float(format(float(LEAST_GMEM / Gall), '.2f'))
        if int(GPUtil.getAvailability([GPU], maxLoad=GPU_LAOD, maxMemory=GMEM_LAOD_LIMIT)) == 1:
            AVAIL_DEVICE_LIST.append(GPU)
            if GMEM_LAOD_LIMIT < MIN_FRAC:
                GMEM_LAOD_LIMIT = MIN_FRAC
            if GMEM_LAOD_LIMIT > MAX_FRAC:
                GMEM_LAOD_LIMIT = MAX_FRAC
            AVAIL_DEVICE_MEMFRAC.append(GMEM_LAOD_LIMIT)
            AVAIL_DEVICE_MAXTHREAD.append(int(1.0/GMEM_LAOD_LIMIT))
except Exception as e:
    print(e)

# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

BATCH_SIZE = 1
# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
TEXT_QUEUE = "text_queue"
SERVER_SLEEP = 0.1
CLIENT_SLEEP = 0.1

# Output Throttle
THROTTLE = 0.9

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
