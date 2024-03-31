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
from google.colab.patches import cv2_imshow
warnings.filterwarnings("ignore")


pathway = "/Users/andrey/Downloads/archive"
data_df = pd.read_csv(os.path.join(pathway, "overview.csv"))

# print("Number of TIFF Images:", len(os.listdir(os.path.join(pathway, "tiff_images"))))
tiff_data = pd.DataFrame([{'path': filepath} for filepath in glob(pathway + '/tiff_images/*.tif')])


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


# a function that read an image
def readImg(img):
    # Load the input image
    image = imread(img)
    return image


dicomImg = []
for i in range(len(dicom_data)):
    dicomImg.append(readImg(dicom_data["Path"][i]))


# print('Original Dimensions : ', dicomImg[0].shape)

image_size = 256
dim = (image_size, image_size)

# resize image
resized = cv2.resize(dicomImg[0], dim, interpolation=cv2.INTER_CUBIC)

# print('Resized Dimensions : ', resized.shape)

# cv2_imshow(resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2_imshow()

tiffImg = []
for i in range(len(tiff_data)):
    tiffImg.append(readImg(tiff_data["Path"][i]))


# print('Original Dimensions : ', tiffImg[0].shape)


image_size = 256
dim = (image_size, image_size)

# resize image
resized = cv2.resize(tiffImg[0], dim, interpolation=cv2.INTER_CUBIC)

# print('Resized Dimensions : ', resized.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()


def resize(img):
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('Resized', resized)
    #cv2.waitKey(0)
    return resized


resizedImg = []
resizedImg_tiff = []
for i in range(len(dicom_data)):
    resizedImg.append(resize(dicomImg[i]))
    resizedImg_tiff.append(resize(tiffImg[i]))

# print(resizedImg[0].shape)


def normalize(img):
    normalized = cv2.normalize(img, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
    return normalized


normalizedImg = []
normalizedImg_tiff = []
for i in range(len(dicom_data)):
    normalizedImg.append(normalize(resizedImg[i]))
    normalizedImg_tiff.append(normalize(resizedImg_tiff[i]))


shapedData = []
shapedData_tiff = []
for i in range(len(dicom_data)):
    shapedData.append(normalizedImg[i].reshape(-1)/ 255.0)
    shapedData_tiff.append(normalizedImg_tiff[i].reshape(-1)/ 255.0)

print(shapedData[0])