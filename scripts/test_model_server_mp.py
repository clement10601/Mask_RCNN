import base64
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('PATH', '')
import numpy as np
import config
import time
import json
from io import BytesIO
from multiprocessing import Process, Pipe, current_process, Lock, Queue
import GPUtil
from skimage.measure import find_contours
import skimage.io
import random
import datetime

IMG_DIR = '/tmp/images'
file_names = []
files = list(f for f in os.listdir(IMG_DIR) 
        if os.path.isfile(os.path.join(IMG_DIR, f)))
imageStack = []
for f in files:
    f_name = os.path.join(IMG_DIR, f)
    imageStack.append(skimage.io.imread(f_name))
for i in range(1000):
    file_names.append(random.randint(0, len(files)-1))
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
    def __init__(self, queue, LOCK, GPU="", FRAC=0, name='0'):
        Process.__init__(self)
        self.lock = LOCK
        self.queue = queue
        self.name = name
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
        from mrcnn.tools import utils
        from mrcnn.maskrcnn import model as modellib
        from mrcnn.maskrcnn import stdinstance
        from PIL import Image
        self.Image = Image
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
                query = self.queue.get(timeout=1)
                imageIDs = []
                batch = []
                # loop over the queue
                # deserialize the object and obtain the input image
                image = imageStack[query]
                # check to see if the batch list is None
                batch.append(image)
                # update the list of image IDs
                if self.name == '0':
                    print(datetime.datetime.fromtimestamp(time.time()))
                # check to see if we need to process the batch
                if len(batch) > 0:
                    #print('{}: Procesing {} images!'.format(self.name, len(imageIDs)))
                    start = time.time()
                    with graph.as_default():
                        results = model.detect(batch, verbose=config.VERBOSE)
                    end = time.time()
                    et = end - start
                    self.dt += float(et)
                    self.counter += 1
                    print(et)
                    # loop over the image IDs and their corresponding set of
                    # results from our model
                # sleep for a small amount
                time.sleep(config.SERVER_SLEEP)
            except Exception as e:
                print(e)
                time.sleep(config.SERVER_SLEEP)
                break
        adt = float(self.dt)/float(self.counter)
        print('avg dt: %f' % adt) 
        print('%s proc, cont: %d' % (self.name, self.counter)) 
        return

if __name__ == "__main__":
    LOCK = Lock()
    AVAIL_DEVICE_LIST = config.AVAIL_DEVICE_LIST
    AVAIL_DEVICE_MEMFRAC = config.AVAIL_DEVICE_MEMFRAC
    AVAIL_DEVICE_MAXTHREAD = config.AVAIL_DEVICE_MAXTHREAD
    q = Queue()
    for item in file_names:
        q.put(item)
    
    exest = time.time()
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
                p = mlWorker(q, LOCK, GPU=device, FRAC=mem_frac, name=str(thread))
                p.daemon = True
                proc_list.append(p)
        print('Starting total: {} processes'.format(len(proc_list)))
        for proc in proc_list:
            proc.start()
        print('All processes started')
    else:
        p = mlWorker(q, LOCK)
        p.daemon = True
        p.start()
        p.join()

    if proc_list:
        for proc in proc_list:
            proc.join()
    exend = time.time()
    print("Total: {}".format(exend - exest))
