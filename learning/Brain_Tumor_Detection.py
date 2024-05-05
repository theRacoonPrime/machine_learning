import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm  # Importing tqdm for progress bars
import tensorflow as tf

BATCH_SIZE = 4  # Batch size for training
GLOBAL_CLIPNORM = 10.0  # Global clipnorm value for gradient clipping

AUTO = tf.data.AUTOTUNE  # Autotune parameter for performance optimization

# Function to parse text annotations
def parse_txt_annot(img_path, txt_path):
    img = cv2.imread(img_path)
    w = int(img.shape[0])
    h = int(img.shape[1])

    file_label = open(txt_path, "r")
    lines = file_label.read().split('\n')

    boxes = []
    classes = []

    if lines[0] == '':
        return img_path, classes, boxes
    else:
        for i in range(0, int(len(lines))):
            objbud = lines[i].split(' ')
            class_ = int(objbud[0])

            x1 = float(objbud[1])
            y1 = float(objbud[2])
            w1 = float(objbud[3])
            h1 = float(objbud[4])

            xmin = int((x1 * w) - (w1 * w) / 2.0)
            ymin = int((y1 * h) - (h1 * h) / 2.0)
            xmax = int((x1 * w) + (w1 * w) / 2.0)
            ymax = int((y1 * h) + (h1 * h) / 2.0)

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(class_)

    return img_path, classes, boxes

# Function to create a list of file paths
def create_paths_list(path):
    full_path = []
    images = sorted(os.listdir(path))

    for i in images:
        full_path.append(os.path.join(path, i))

    return full_path

# Mapping class ids to class labels
class_ids = ['label0', 'label1', 'label2']
class_mapping = dict(zip(range(len(class_ids)), class_ids))

print(class_mapping)


