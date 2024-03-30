IS_LOCAL = False  # Define IS_LOCAL flag here

import numpy as np
import pandas as pd
from skimage.io import imread
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import os
import dicom

if IS_LOCAL:
    import pydicom as dicom
else:
    import dicom


if IS_LOCAL:
    PATH = "/Users/andrey/Downloads/archive"
else:
    PATH = "/Users/andrey/Downloads/archive"
print(os.listdir(PATH))

data_df = pd.read_csv(os.path.join(PATH, "overview.csv"))

tiff_data = pd.DataFrame([{'path': filepath} for filepath in glob(os.path.join(PATH, 'tiff_images/*.tif'))])


def process_data(path):
    data = pd.DataFrame([{'path': filepath} for filepath in glob(os.path.join(PATH, path))])
    data['file'] = data['path'].map(os.path.basename)
    data['ID'] = data['file'].map(lambda x: str(x.split('_')[1]))
    data['Age'] = data['file'].map(lambda x: int(x.split('_')[3]))
    data['Contrast'] = data['file'].map(lambda x: bool(int(x.split('_')[5])))
    data['Modality'] = data['file'].map(lambda x: str(x.split('_')[6].split('.')[-2]))

    # Print data to check its structure and column names
    print(data.head())

    return data


# Check the structure of the DICOM data DataFrame
dicom_data = process_data('dicom_dir/*.dcm')


def countplot_comparison(feature):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16, 4))
    s1 = sns.countplot(data_df[feature], ax=ax1)
    s1.set_title("Overview data")
    s2 = sns.countplot(tiff_data[feature], ax=ax2)
    s2.set_title("Tiff files data")
    s3 = sns.countplot(dicom_data[feature], ax=ax3)
    s3.set_title("Dicom files data")
    plt.show()




