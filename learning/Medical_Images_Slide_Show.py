import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
from skimage.io import imread
import dicom
import warnings
from list_of_defs import (countplot_comparison, normalize)
warnings.filterwarnings("ignore")

# Path to data
pathway = "/Users/andrey/Downloads/archive"

# Read overview data
data_df = pd.read_csv(os.path.join(pathway, "overview.csv"))

# Function to process image data


def process_data(path):
    data = pd.DataFrame([{'Path': filepath} for filepath in glob(pathway + path)])
    data['File'] = data['Path'].map(os.path.basename)
    data['ID'] = data['File'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['File'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['File'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['File'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return data

# Process TIFF image data


tiff_data = process_data('/tiff_images/*.tif')

# Process DICOM image data
dicom_data = process_data('/dicom_dir/*.dcm')

# Function to read image
def readImg(img):
    # Load the input image
    image = imread(img)
    return image

# Read DICOM images


dicomImg = [readImg(path) for path in dicom_data["Path"]]

# Read TIFF images
tiffImg = [readImg(path) for path in tiff_data["Path"]]

# Image resizing
image_size = 256
dim = (image_size, image_size)


def resize(img):
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return resized

# Resize DICOM images


resizedImg = [resize(img) for img in dicomImg]


# Resize TIFF images
resizedImg_tiff = [resize(img) for img in tiffImg]

# Normalize DICOM images

normalizedImg = [normalize(img) for img in resizedImg]

# Normalize TIFF images
normalizedImg_tiff = [normalize(img) for img in resizedImg_tiff]

# Reshape and normalize image data
shapedData = [img.reshape(-1)/ 255.0 for img in normalizedImg]
shapedData_tiff = [img.reshape(-1)/ 255.0 for img in normalizedImg_tiff]


