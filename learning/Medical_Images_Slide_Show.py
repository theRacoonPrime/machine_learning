IS_LOCAL = False
import numpy as np
import pandas as pd
from skimage.io import imread
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os
import dicom

if(IS_LOCAL):
    import pydicom as dicom
else:
    import dicom


if(IS_LOCAL):
    PATH = "/Users/andrey/Downloads/archive"
else:
    PATH = "/Users/andrey/Downloads/archive"
print(os.listdir(PATH))

data_df = pd.read_csv(os.path.join(PATH, "overview.csv"))

# print("CT Medical images -  rows:", data_df.shape[0], " columns:", data_df.shape[1])
# print(data_df.head())

# print("Number of TIFF images:", len(os.listdir(os.path.join(PATH, "tiff_images"))))


