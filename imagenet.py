import os
import time
import numpy as np
import sys
import zipfile
import urllib.request
import shutil
from tools.config import Config
from tools import utils
import maskrcnn.model as modellib
import xml, xml.dom.minidom

# Root directory of the project
ROOT_DIR = os.getcwd()
# Path to trained weights file
IMGNET_MODEL_PATH = os.path.join(ROOT_DIR, 
    "weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class ImagenetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "imagenet"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class ImagenetDataset(utils.Dataset):
    def load_imagenet(self, ann_dir, img_dir, class_ids, label_file):
        class_ids = None
        image_ids = None
        if not class_ids:
            class_ids = os.listdir(ann_dir)
        
        label = {}
        with open(label_file) as f:
            for line in f:
                line = line.strip()
                b = line.split()
                label[b[0]] = b[1] 
        for i in class_ids:
            self.add_class("imagenet", i, label[i])
        
        for annid in class_ids:
            pth = os.path.join(ann_dir, annid)
            img_pth = os.path.join(img_dir, annid)
            assert os.path.isdir(pth)
            files = [f for f in os.listdir(pth) if os.path.isfile(f)]
            for xmlfile in files:
                imgdict = {}
                try:
                    DomTree = xml.dom.minidom.parse(os.path.join(pth, xmlfile))  
                    annotation = DomTree.documentElement
                    # Extract top lists from root annotation
                    filenamelist = annotation.getElementsByTagName('filename')
                    sizelist = annotation.getElementsByTagName('size')
                    seglist = annotation.getElementsByTagName('segmented')
                    objectlist = annotation.getElementsByTagName('object')
                    
                    #print(filenamelist[0].childNodes[0].data)
                    imgdict['imgid'] = filenamelist[0].childNodes[0].data
                    imgdict['filename'] = imgdict['imgid'] + '.jpeg'
                    imgdict['filepath'] = str(os.path.join(img_pth, imgdict['filename']))

                    # Extract 2nd lvl lists from top lists
                    for specs in sizelist:
                        widthDOM = specs.getElementsByTagName('width')
                        heightDOM = specs.getElementsByTagName('height')
                        depthDOM = specs.getElementsByTagName('depth')

                        width = widthDOM[0].childNodes[0].data
                        height = heightDOM[0].childNodes[0].data
                        depth = depthDOM[0].childNodes[0].data
                        #print('{} {} {}'.format(width, height, depth))
                        imgdict['spec'] = {}
                        imgdict['spec']['width'] = width
                        imgdict['spec']['height'] = height
                        imgdict['spec']['depth'] = depth

                    # Find seg count
                    seg_count = int(seglist[0].childNodes[0].data)
                    imgdict['segs'] = seg_count
                    imgdict['annotations'] = []
                    for objects in objectlist:
                        instance = {}
                        namelist = objects.getElementsByTagName('name')
                        objectname = namelist[0].childNodes[0].data
                        #print(objectname)
                        instance['name'] = objectname
                        instance['bbox'] = []
                        bndbox = objects.getElementsByTagName('bndbox')
                        bc = 0
                        for box in bndbox:
                            bbox = []
                            try:
                                x1_list = box.getElementsByTagName('xmin')
                                x1 = int(x1_list[0].childNodes[0].data)
                                y1_list = box.getElementsByTagName('ymin')
                                y1 = int(y1_list[0].childNodes[0].data)
                                x2_list = box.getElementsByTagName('xmax')
                                x2 = int(x2_list[0].childNodes[0].data)
                                y2_list = box.getElementsByTagName('ymax')
                                y2 = int(y2_list[0].childNodes[0].data)
                                w = x2 - x1
                                h = y2 - y1
                                #print('{} {} {} {}'.format(x1, y1, w, h))
                                bbox.append(x1)
                                bbox.append(y1)
                                bbox.append(w)
                                bbox.append(h)
                                instance['bbox'].append(bbox)
                                annotation = {}
                                # annotation['segmentation'] = [self.getsegmentation()]
                                annotation['segmentation'] = ''
                                annotation['iscrowd'] = 0
                                annotation['image_id'] = imgdict['imgid']
                                # annotation['bbox'] = list(map(float, self.bbox))
                                annotation['bbox'] = bbox
                                annotation['category_id'] = annid
                                annotation['id'] = bc
                                bc+=1
                                imgdict['annotations'].append(annotation)
                            except Exception as e:
                                print(e)
                    #######################
                    self.add_image(
                        "imagenet", image_id=imgdict['imgid'],
                        path=imgdict['filepath'],
                        width=imgdict['spec']['width'],
                        height=imgdict['spec']['height'],
                        annotations=imgdict['annotations'])
                except Exception as e:
                    print(e)
    # overwrite
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ImagenetDataset, self).load_mask(image_id)
    # overwrite
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(ImagenetDataset, self).image_reference(image_id)

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        bbox = ann['bbox']
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
