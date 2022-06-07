import os
import sys
import random
import math
import cv2
import numpy as np
import pyttsx3
from typing import NamedTuple
import skimage.io
from skimage.measure import find_contours
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
from multiprocessing import Process
from threading import Thread

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.coco import coco

# Root directory of the project
ROOT_DIR = os.path.abspath("/Users/judy/Documents/UW/Courses/CSE576/Project/Mask_RCNN") # ("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Target(NamedTuple):
    label: str
    roi: list
    score: float
    width: int
    height: int
    area: int

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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
traffic_names = ['person', 'bicycle', 'car', 'motorcycle', 
               'bus', 'train', 'truck', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter']

global F
F = 0
WIDTH_THRESDHOLD = 50
HEIGHT_THRESDHOLD = 100
AREA_THRESDHOLD = 10000

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(10, 10), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):

        if class_names[class_ids[i]] not in traffic_names:
            continue

        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    # if auto_show:
    # plt.show()

    plt.draw()
    plt.pause(0.0001)
    plt.clf()

def process_results(r, image):
    height = image.shape[0]
    width = image.shape[1]
    person_height = 16500
    focal_length = 28
    ids = r['class_ids']
    rois = r['rois']
    scores = r['scores']
    num_detection = len(ids)
    targets = []
    for i in range(num_detection):
        label = class_names[ids[i]]
        roi = rois[i]
        score = scores[i]
        area = (roi[2] - roi[0]) * (roi[3] - roi[1])
        # If on the edge, skip as it can be outlier
        if roi[1] == 0 or roi[3] == width:
            continue
        if (label in traffic_names):
            target = Target(label, roi, score, roi[3] - roi[1], roi[2] - roi[0], area)
            targets += [target]
            print(label, roi, score, area)
    targets = sorted(targets, key=lambda x: x.area, reverse=True)
    global F
    if F < 0.0001 and len(targets) > 0:
        if targets[-1].label == "person" and targets[-1].height <= HEIGHT_THRESDHOLD:
            F = targets[-1].height * 400.0 / 65.0
            print(F)
    return targets

def output(targets, image):
    threshold = 100
    width = image.shape[1]
    text = ""
    short_text = ""
    count = 0

    for idx, target in enumerate(targets):
        label = target.label
        area = target.area
        direction = ""
        distance = 10000
        if F > 0.0001 and label == "person":
            distance = 25.0 * F / target.width
            if distance < threshold:
                count += 1
        if target.roi[3] < width / 2.0:
            direction = "to your left"
        elif target.roi[1] > width / 2.0:
            direction = 'to your right'
        else:
            direction = "in front of you"

        if idx == 0 and (distance < threshold or area > AREA_THRESDHOLD):
            text = text + "\nThere's a {} {}".format(label, direction)
            short_text = short_text + label + " "
            if F > 0.0001 and label == "person" :
                text = text + ", about {} inches from you".format(int(distance))
                short_text = short_text + str(int(distance)) + " inches "
            text = text + ". "
            short_text = short_text + direction
    if count >= 3:
        text = text + "\nThere's a crowd in front of you. "
        short_text = short_text + " and crowd"
    if text == "":
        text = "All clear!"
    else: 
        text = text + "Be careful! "
    return text, short_text

class _TTS:

    engine = None
    rate = None
    def __init__(self):
        self.engine = pyttsx3.init()

    def start(self,text_):
        self.engine.say(text_)
        self.engine.runAndWait()

def speak(short_text):
    # engine = pyttsx3.init()
    # engine.say(short_text)
    # engine.runAndWait()

    tts = _TTS()
    tts.start(short_text)
    del(tts)

def run():
    """
    Construct and show the Toga application.

    Usually, you would add your application to a main content box.
    We then create a main window (with a name matching the app), and
    show the main window.
    """

    # Load a random image from the images folder
    # image = skimage.io.imread(os.path.join(ROOT_DIR, "street_view.mp4"))
    vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "street_view.mp4"))
    success, image = vidcap.read()
    skip_frame = 200 # 36
    count = 0
    while True:
        if (count % skip_frame == 0):
            results = model.detect([image], verbose=1)
            r = results[0]
            targets = process_results(r, image)
            text, short_text = output(targets, image)
            display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                class_names, r['scores'], title=text)
            if short_text != "":
                try:
                    p = Thread(target=speak, args=(short_text,))
                    p.start()
                except:
                    pass
                
        success, image = vidcap.read()
        count += 1

# run()
vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "street_view.mp4"))
success, image = vidcap.read()
skip_frame = 24
count = 0
while True:
    if (count % skip_frame == 0):
        results = model.detect([image], verbose=1)
        r = results[0]
        targets = process_results(r, image)
        text, short_text = output(targets, image)
        display_instances(image, r['rois'], r['masks'], r['class_ids'], 
            class_names, r['scores'], title=text)
        if short_text != "":
            tts = _TTS()
            tts.start(short_text)
            del(tts)
    success, image = vidcap.read()
    count += 1

