import base64
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('PATH', '')
import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import config
import redis
import time
import json
from PIL import Image
from io import BytesIO

from mrcnn.tools import utils
from mrcnn.tools.config import Config
from mrcnn.maskrcnn import model as modellib
from mrcnn.maskrcnn import stdinstance

# connect to Redis server
redisDB = redis.StrictRedis(host=config.REDIS_HOST,
                            port=config.REDIS_PORT,
                            db=config.REDIS_DB,
                            socket_timeout=1,
                            socket_keepalive=True)

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def extract_result(boxes, masks, class_ids, class_names,
                      scores=None, title="", throttle='0.95'):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    return: [{'label': label, 'score': score}, {}, ...]
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    output = []
    for i in range(N):
        score = scores[i] if scores is not None else None
        if float(score) < float(throttle):
            continue
        # Label
        class_id = class_ids[i]
        label = class_names[class_id]
        out = {'label': label, 'score': float(score)}
        output.append(out)
    return output


def base64_decode_image(a):
    '''
    return: <ndarray>
    '''
    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    #if sys.version_info.major == 3:
    #    a = bytes(a, encoding="utf-8")
    # io = BytesIO(a)
    # img = Image.open(io)
    # img = np.array(img)
    img = Image.open(BytesIO(base64.b64decode(a)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = np.array(img)
    return img


def classify_process():
    print('ML Process initialing...')
    # Root directory of the project
    ROOT_DIR = config.ROOT_DIR
    # imgfiles
    UPLOAD_FOLDER = config.UPLOAD_FOLDER
    # Directory to save logs and trained model
    MODEL_DIR = config.MODEL_DIR
    # Local path to trained weights file
    COCO_MODEL_DIR = config.COCO_MODEL_DIR
    COCO_MODEL_PATH = config.COCO_MODEL_PATH
    intconf = InferenceConfig()
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
    keras.backend.clear_session()
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = config.MEM_FRAC
    set_session(tf.Session(config=conf))
    # Create model object in inference mode.
    _model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=intconf)
    # Load weights trained on MS-COCO
    _model.load_weights(COCO_MODEL_PATH, by_name=True)
    graph = tf.get_default_graph()
    print('ML Process Runnung...')

    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        try:
            queue = redisDB.lrange(config.IMAGE_QUEUE, 0,
                              config.BATCH_SIZE - 1)
            imageIDs = []
            batch = []
            if queue:
                pass
            else:
                time.sleep(config.SERVER_SLEEP)
                continue

            # loop over the queue
            for q in queue:
                # deserialize the object and obtain the input image
                data = json.loads(q)
                image = base64_decode_image(data["image"])
                # check to see if the batch list is None
                batch.append(image)
                # update the list of image IDs
                imageIDs.append(data["id"])

            # check to see if we need to process the batch
            if len(imageIDs) > 0:
                print('New job received')
                with graph.as_default():
                    results = _model.detect(batch, verbose=config.VERBOSE)
                # loop over the image IDs and their corresponding set of
                # results from our model
                for (imageID, resultSet) in zip(imageIDs, results):
                    # initialize the list of output predictions
                    output = []
                    output = extract_result(resultSet['rois'], resultSet['masks'],
                        resultSet['class_ids'], class_names, resultSet['scores'], 
                        throttle=config.THROTTLE)
                    redisDB.set(imageID, json.dumps(output))
                # remove the set of images from our queue
                redisDB.ltrim(config.IMAGE_QUEUE, len(imageIDs), -1)
            # sleep for a small amount
            time.sleep(config.SERVER_SLEEP)
        except Exception as e:
            pass


if __name__ == "__main__":
    classify_process()
