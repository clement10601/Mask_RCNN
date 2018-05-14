#coding:utf-8
import base64
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('PATH', '')
import numpy as np
import time
import config
import redis
import json
from io import BytesIO
from multiprocessing import Process, Pipe, current_process, Lock
import GPUtil


# connect to Redis server
redisDB = redis.StrictRedis(host=config.REDIS_HOST,
                          port=config.REDIS_PORT,
                          db=config.REDIS_DB,
                          socket_timeout=1,
                          socket_keepalive=True)


class mlWorker(Process):
    def __init__(self, LOCK, GPU=None, FRAC=0):
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

    def run(self):
        from cocr import model
        from PIL import Image
        self.Image = Image
        if self.GPU:
            print('ML Process: {} starting, using GPU: {}, frac: {}'.format(self.name,self.GPU.id,self.frac))
        _model = model
        self.mainloop(model=_model)

    def mainloop(self, model='', graph=''):
        while True:
            # attempt to grab a batch of images from the database, then
            # initialize the image IDs and batch of images themselves
            try:
                redisDB = redis.StrictRedis(host=config.REDIS_HOST,
                          port=config.REDIS_PORT,
                          db=config.REDIS_DB,
                          socket_timeout=1,
                          socket_keepalive=True)
                self.lock.acquire()
                query = redisDB.lpop(config.TEXT_QUEUE)
                self.lock.release()
                imageIDs = []
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

                # check to see if we need to process the batch
                if len(imageIDs) > 0:
                    print('{}: Run Task!'.format(self.name))
                    result, img, angle = model.model(image, model='keras')
                    # loop over the image IDs and their corresponding set of
                    # results from our model
                    output = {}
                    for key in result:
                        output[key]=result[key][1]
                    for imageID in imageIDs:
                        redisDB.set(imageID, json.dumps(output))
                # sleep for a small amount
                time.sleep(config.SERVER_SLEEP*2)
            except Exception as e:
                print(e)
                time.sleep(config.SERVER_SLEEP)
                continue

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
