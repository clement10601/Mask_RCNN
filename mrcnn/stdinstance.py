"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
############################################################
#  Std out funcs
############################################################
def extract_instances(boxes, masks, class_ids, class_names,
                      scores=None, title="", score_throttle='0.95'):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
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
        if float(score) < float(score_throttle):
            continue
        # Label
        class_id = class_ids[i]
        label = class_names[class_id]
        out = {label}
        output.append(label)
    return output