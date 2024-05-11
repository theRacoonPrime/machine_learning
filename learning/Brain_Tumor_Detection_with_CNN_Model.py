import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import random

path = '/Users/andrey/Downloads/brain_mri_scan_images 2'
data_path = path.split(os.sep)

main_path = os.getcwd().split(os.sep)

os.listdir(os.sep.join(data_path))


positive_dir = data_path + ['positive']
negative_dir = data_path + ['negative']


random_positive_images = random.sample(os.listdir(os.sep.join(positive_dir)), 2)
random_negative_images = random.sample(os.listdir(os.sep.join(negative_dir)), 2)


pos_img_1 = plt.imread(os.sep.join(positive_dir + [random_positive_images[0]]))
pos_img_2 = plt.imread(os.sep.join(positive_dir + [random_positive_images[1]]))
neg_img_1 = plt.imread(os.sep.join(negative_dir + [random_negative_images[0]]))
neg_img_2 = plt.imread(os.sep.join(negative_dir + [random_negative_images[1]]))
