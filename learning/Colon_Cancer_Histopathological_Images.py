import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# import cv2
from PIL import Image
import os
import glob as gb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam , lr_scheduler
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
from sklearn.model_selection import train_test_split


def walk_through_data(dir_path) :
    for dirpath , dirnames , filenames in tqdm(os.walk(dir_path)) :
        print(f"There are {len(dirnames)} directions and {len(filenames)} images in {dirpath}")


dataset_path='/Users/andrey/Downloads/lung_colon_image_set'
pred_path = 'lung_colon_image_prediction_set'

walk_through_data(dataset_path)
walk_through_data(pred_path)


extension=[]
for cat in tqdm(os.listdir(dataset_path)) :
    for folder in os.listdir(dataset_path + "/" + cat) :
        for file in os.listdir(dataset_path + "/" + cat + "/" + folder + "/") :
            if os.path.isfile(dataset_path + "/" + cat + "/" + folder + "/" + file) :
                extension.append(os.path.splitext(file)[1])

print(len(extension),np.unique(extension))

categories = []
classes = []
for cat in tqdm(os.listdir(dataset_path)):
    if cat not in categories:
        categories.append(cat.replace('_image_sets', ''))

    for folder in os.listdir(dataset_path + "/" + cat):
        if folder not in classes:
            classes.append(folder)


img_label={}
for key in categories :
    dic={}
    for value in classes :
        if key in value :
            dic[value]=classes.index(value)
    img_label[key]=dic


def getlabel(n) :
    for i, j in img_label.items() :
        for x, y in j.items() :
            if n==y :
                return i, x


num_of_disease = {}

for cat in tqdm(os.listdir(dataset_path)):
    for folder in os.listdir(dataset_path + "/" + cat):
        num_of_disease[folder] = len(os.listdir(dataset_path + "/" + cat + "/" + folder))


img_per_class = pd.DataFrame(num_of_disease.values(),
                             index = num_of_disease.keys(),
                             columns=["# of images"]
                            )