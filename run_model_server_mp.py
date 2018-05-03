import base64
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('PATH', '')
import numpy as np
import config
import redis
import time
import json
from io import BytesIO
from multiprocessing import Process, Pipe, current_process, Lock
import GPUtil
from skimage.measure import find_contours


# connect to Redis server
redispool = redis.ConnectionPool(host=config.REDIS_HOST,
                          port=config.REDIS_PORT,
                          db=config.REDIS_DB,
                          socket_keepalive=True)

try:
    print('Testing Redis Connection')
    redisdbSession = redis.StrictRedis(connection_pool=redispool)
    response = redisdbSession.client_list()
    print('Redis Connection Established')
except redis.ConnectionError as e:
    print(e)
    sys.exit(1)

# Root directory of the project
ROOT_DIR = config.ROOT_DIR
# imgfiles
UPLOAD_FOLDER = config.UPLOAD_FOLDER
# Directory to save logs and trained model
MODEL_DIR = config.MODEL_DIR
# Local path to trained weights file
COCO_MODEL_DIR = config.COCO_MODEL_DIR
COCO_MODEL_PATH = config.COCO_MODEL_PATH
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


class mlWorker(Process):
    def __init__(self, LOCK, GPU="", FRAC=0):
        Process.__init__(self)
        self.lock = LOCK
        if GPU:
            print('{} using GPUid: {}, Name: {}'.format(self.name, str(GPU.id), str(GPU.name)))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU.id)
            self.device = '/device:GPU:0'
        else:
            self.device = ''
        self.GPU = GPU
        self.frac = FRAC
        self.counter = 0
        self.dt = 0.0

    def run(self):
        import keras
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf
        from mrcnn import model as modellib
        from PIL import Image
        self.Image = Image
        if self.GPU:
            print('ML Process: {} starting, using GPU: {}, frac: {}'.format(self.name,self.GPU.id,self.frac))
        keras.backend.clear_session()
        conf = tf.ConfigProto()
        conf.gpu_options.per_process_gpu_memory_fraction = self.frac
        set_session(tf.Session(config=conf))

        # Create model object in inference mode.
        intconf = config.InferenceConfig()
        _model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=intconf)
        # Load weights trained on MS-COCO
        _model.load_weights(COCO_MODEL_PATH, by_name=True)
        graph = tf.get_default_graph()
        print('ML Process: {} started'.format(self.name))
        self.mainloop(model=_model, graph=graph)

    def mainloop(self, model='', graph=''):
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            try:
                redisdbSession = redis.StrictRedis(connection_pool=redispool)
                self.lock.acquire()
                query = redisdbSession.lrange(config.IMAGE_QUEUE, 0, config.BATCH_SIZE - 1)
                redisdbSession.ltrim(config.IMAGE_QUEUE, len(query), -1)
                self.lock.release()
                imageIDs = []
                thresholds = {}
                batch = []
                # loop over the queue
                # deserialize the object and obtain the input image
                if query:
                    for item in query:
                        data = json.loads(item)
                        image = self.base64_decode_image(data["image"])
                        # check to see if the batch list is None
                        batch.append(image)
                        # update the list of image IDs
                        imageIDs.append(data["id"])
                        thresholds[data["id"]] = data["threshold"]

                # check to see if we need to process the batch
                if len(imageIDs) > 0:
                    #print('{}: Procesing {} images!'.format(self.name, len(imageIDs)))
                    start = time.time()
                    with graph.as_default():
                        results = model.detect(batch, verbose=config.VERBOSE)
                    end = time.time()
                    et = end - start
                    self.dt += float(et)
                    self.counter += 1
                    adt = float(self.dt)/float(self.counter)
                    print('avg dt: %f' % adt) 
                    # loop over the image IDs and their corresponding set of
                    # results from our model
                    for (imageID, resultSet) in zip(imageIDs, results):
                        # initialize the list of output predictions
                        output = []
                        output = self.extract_result(resultSet['rois'], resultSet['masks'],
                            resultSet['class_ids'], class_names, resultSet['scores'],
                            throttle=float(thresholds[imageID]))
                        redisdbSession.set(imageID, json.dumps(output))
                # sleep for a small amount
                time.sleep(config.SERVER_SLEEP*2)
            except Exception as e:
                print(e)
                time.sleep(config.SERVER_SLEEP)
                continue

    def extract_result(self, boxes, masks, class_ids, class_names,
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
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            # bbox
            y1, x1, y2, x2 = boxes[i]
            # label
            class_id = class_ids[i]
            # mask
            mask = masks[:, :, i]
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            pol = []
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                pol.append(verts.tolist())

            label = class_names[class_id]
            out = {'label': label, 
                   'score': float(score), 
                   'bbox': [np.asscalar(x1), np.asscalar(y1), np.asscalar(x2-x1), np.asscalar(y2-y1)],
                   'mask': pol}
            output.append(out)
        return output

    def base64_decode_image(self, a):
        """
        return: <ndarray>
        """
        img = self.Image.open(BytesIO(base64.b64decode(a)))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img)
        return img



if __name__ == "__main__":
    LOCK = Lock()
    AVAIL_DEVICE_LIST = config.AVAIL_DEVICE_LIST
    AVAIL_DEVICE_MEMFRAC = config.AVAIL_DEVICE_MEMFRAC
    AVAIL_DEVICE_MAXTHREAD = config.AVAIL_DEVICE_MAXTHREAD

    proc_list = []
    print('{} GPUs Available'.format(len(AVAIL_DEVICE_LIST)))
    if AVAIL_DEVICE_LIST:
        for index, device in enumerate(AVAIL_DEVICE_LIST):
            thread_count = int(AVAIL_DEVICE_MAXTHREAD[index])
            mem_frac = float(AVAIL_DEVICE_MEMFRAC[index])
            if config.MAX_FRAC < mem_frac:
                mem_frac = config.MAX_FRAC
            print('Preparing {} process on GPU: {}, frac: {}'.format(thread_count, device.id, mem_frac))
            if config.MAX_THREADS < thread_count:
                thread_count = config.MAX_THREADS
            for thread in range(thread_count):
                p = mlWorker(LOCK, GPU=device, FRAC=mem_frac)
                p.daemon = True
                proc_list.append(p)
        print('Starting total: {} processes'.format(len(proc_list)))
        for proc in proc_list:
            proc.start()
        print('All processes started')
    else:
        p = mlWorker(LOCK)
        p.daemon = True
        p.start()
        p.join()

    if proc_list:
        for proc in proc_list:
            proc.join()
