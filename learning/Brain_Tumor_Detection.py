import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm  # Importing tqdm for progress bars
import tensorflow as tf
from list_of_defs import  parse_txt_annot

BATCH_SIZE = 4  # Batch size for training
GLOBAL_CLIPNORM = 10.0  # Global clipnorm value for gradient clipping

AUTO = tf.data.AUTOTUNE  # Autotune parameter for performance optimization

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


