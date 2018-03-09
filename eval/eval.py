
import os
import sys
import numpy as np

sys.path.append('/home/john/Mask_RCNN/')
sys.path.append('/home/john/Mask_RCNN/maskrcnn/')
sys.path.append('/home/john/Mask_RCNN/cocoapi/PythonAPI/')

import utils


# MS COCO Dataset
import coco
print('coco lib path:{}'.format(coco.__file__))
config = coco.CocoConfig()
# define path of coco dataset
COCO_DIR = "/data1/coco/2017"

# Load dataset
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

for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    print('----------------------------------')
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    #visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
    
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)
    
    # Show image additional info
    print("image_id: ", image_id, dataset.image_reference(image_id))
    print(class_ids)

    for no, i in enumerate(class_ids):
        print("class_ids:", i)
        print("class_name", dataset.class_names[i])
        print("bbox:", bbox[no]) 


