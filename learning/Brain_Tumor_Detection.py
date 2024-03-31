import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras
import cv2
from sklearn.model_selection import train_test_split
import keras_cv


path_image = "/kaggle/input/medical-image-dataset-brain-tumor-detection/Brain Tumor " \
             "Detection/test/images/volume_100_slice_47_jpg.rf.5a4036c4db721c7a2501e756d91915a6.jpg "
path_label = "/kaggle/input/medical-image-dataset-brain-tumor-detection/Brain Tumor " \
             "Detection/test/labels/volume_100_slice_47_jpg.rf.5a4036c4db721c7a2501e756d91915a6.txt "

