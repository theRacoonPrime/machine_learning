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
warnings.filterwarnings("ignore")

pathway = "/Users/andrey/Downloads/archive"
data_df = pd.read_csv(os.path.join(pathway, "overview.csv"))

def process_data(path):
    data = pd.DataFrame([{'Path': filepath} for filepath in glob(pathway + path)])
    data['File'] = data['Path'].map(os.path.basename)
    data['ID'] = data['File'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['File'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['File'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['File'].map(lambda x: str(x.split('_')[6].split('.')[-2]))
    return data

tiff_data = process_data('/tiff_images/*.tif')
dicom_data = process_data('/dicom_dir/*.dcm')

def countplot_comparison(feature):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    s1 = sns.countplot(data_df[feature], ax=ax1)
    s1.set_title("Overview Data")

    s2 = sns.countplot(tiff_data[feature], ax=ax2)
    s2.set_title("Tiff Images")

    s3 = sns.countplot(dicom_data[feature], ax=ax3)
    s3.set_title("Dicom Images")

    plt.show()

def readImg(img):
    # Load the input image
    image = imread(img)
    return image

dicomImg = [readImg(path) for path in dicom_data["Path"]]
tiffImg = [readImg(path) for path in tiff_data["Path"]]

image_size = 256
dim = (image_size, image_size)

def resize(img):
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return resized

resizedImg = [resize(img) for img in dicomImg]
resizedImg_tiff = [resize(img) for img in tiffImg]

def normalize(img):
    normalized = cv2.normalize(img, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
    return normalized

normalizedImg = [normalize(img) for img in resizedImg]
normalizedImg_tiff = [normalize(img) for img in resizedImg_tiff]

shapedData = [img.reshape(-1)/ 255.0 for img in normalizedImg]
shapedData_tiff = [img.reshape(-1)/ 255.0 for img in normalizedImg_tiff]

