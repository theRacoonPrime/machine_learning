import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
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
    for dirpath, dirnames, filenames in tqdm(os.walk(dir_path)):
        print(f"There are {len(dirnames)} directions and {len(filenames)} images in {dirpath}")


dataset_path = '/Users/andrey/Downloads/lung_colon_image_set'
pred_path = '/Users/andrey/Downloads/lung_colon_image_set'

walk_through_data(dataset_path)
walk_through_data(pred_path)

extension = []
for cat in tqdm(os.listdir(dataset_path)):
    for folder in os.listdir(dataset_path + "/" + cat):
        for file in os.listdir(dataset_path + "/" + cat + "/" + folder + "/"):
            if os.path.isfile(dataset_path + "/" + cat + "/" + folder + "/" + file):
                extension.append(os.path.splitext(file)[1])


categories = []
classes = []
for cat in tqdm(os.listdir(dataset_path)):
    if cat not in categories:
        categories.append(cat.replace('_image_sets', ''))

    for folder in os.listdir(dataset_path + "/" + cat):
        if folder not in classes:
            classes.append(folder)


img_label = {}
for key in categories:
    dic = {}
    for value in classes:
        if key in value:
            dic[value] = classes.index(value)
    img_label[key] = dic


def getlabel(n):
    for i, j in img_label.items():
        for x, y in j.items():
            if n == y:
                return i, x


num_of_disease = {}

for cat in tqdm(os.listdir(dataset_path)):
    for folder in os.listdir(dataset_path + "/" + cat):
        num_of_disease[folder] = len(os.listdir(dataset_path + "/" + cat + "/" + folder))


img_per_class = pd.DataFrame(num_of_disease.values(),
                             index=num_of_disease.keys(), columns=["# of images"])


idx = [i for i in range(len(classes))]
plt.figure(figsize=(20, 10))
plt.bar(idx, [n for n in num_of_disease.values()], width=0.5)
plt.xlabel('Plant/Disease', fontsize=10)
plt.ylabel('# of images')
plt.xticks(idx, classes, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')


dataset_path_list = []
dataset_labels = []
for cat in tqdm(os.listdir(dataset_path)):
    for folder in os.listdir(dataset_path + "/" + cat):
        files = gb.glob(pathname=str(dataset_path + "/" + cat + "/" + folder + "/*.jpeg"))
        for file in files:
            dataset_path_list.append(file)
            dataset_labels.append(img_label[cat.replace('_image_sets', '')][folder])


pred_path_list = []
pred_labels = []
for cat in tqdm(os.listdir(pred_path)):
    for folder in os.listdir(pred_path + "/" + cat):
        files = gb.glob(pathname=str(pred_path + "/" + cat + "/" + folder + "/*.jpeg"))
        for file in files:
            pred_path_list.append(file)
            pred_labels.append(img_label[cat.replace('_image_sets', '')][folder])

train_path_list, test_path_list, train_labels, test_labels = train_test_split(dataset_path_list,
                                                                              dataset_labels, train_size=0.80,
                                                                              random_state=0)
img_size = 250
basic_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor()
])


class InvalidDatasetException(Exception):

    def __init__(self, len_of_paths, len_of_labels):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )


class train_data(Dataset):
    def __init__(self, train_path, train_label, transform_method):
        self.train_path = train_path
        self.train_label = train_label
        self.transform_method = transform_method
        if len(self.train_path) != len(self.train_label):
            raise InvalidDatasetException(self.train_path, self.train_label)

    def __len__(self):
        return len(self.train_path)

    def __getitem__(self, index):
        image = Image.open(self.train_path[index])
        tensor_image = self.transform_method(image)
        label = self.train_label[index]

        return tensor_image, label


train_set = train_data(train_path_list, train_labels, basic_transform)
np.unique(train_set.train_label)
filer_train_image = train_set.__getitem__(1000)
torch.Size([3, 250, 250])
getlabel(filer_train_image[1])


class test_data(Dataset):
    def __init__(self, test_path, test_label, transform_method):
        self.test_path = test_path
        self.test_label = test_label
        self.transform_method = transform_method
        if len(self.test_path) != len(self.test_label):
            raise InvalidDatasetException(self.test_path, self.test_label)

    def __len__(self):
        return len(self.test_path)

    def __getitem__(self, index):
        image = Image.open(self.test_path[index])
        tensor_image = self.transform_method(image)
        label = self.test_label[index]

        return tensor_image, label


test_set = test_data(test_path_list, test_labels, basic_transform)

filer_test_image = test_set.__getitem__(2000)
getlabel(filer_test_image[1])


class pred_data(Dataset):
    def __init__(self, pred_path, transform_method):
        self.pred_path = pred_path
        self.transform = transform_method

    def __len__(self):
        return len(self.pred_path)

    def __getitem__(self, index):
        image = Image.open(self.pred_path[index])
        tensor_image = self.transform(image)

        return tensor_image


pred_set = pred_data(pred_path_list, basic_transform)

filer_pred_image = pred_set.__getitem__(15)
BATCH_SIZE = 128
torch.manual_seed(42)
train_dataloader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

torch.manual_seed(42)
test_dataloader = DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)

torch.manual_seed(42)
pred_dataloader = DataLoader(
    dataset=pred_set,
    batch_size=pred_set.__len__(),
    shuffle=False
)


trainimage_sample, trainlabel_sample = next(iter(train_dataloader))
trainimage_sample.shape, trainlabel_sample.shape


fig, axis = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        img = trainimage_sample[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.set(title=f"{getlabel(trainlabel_sample[i])}")
        ax.axis('off')


testimage_sample , testlabel_sample = next(iter(test_dataloader))


fig, axis = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        img = testimage_sample[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.set(title = f"{getlabel(testlabel_sample[i])}")
        ax.axis('off')


predimage_sample = next(iter(pred_dataloader))
predimage_sample.shape

fig, axis = plt.subplots(3, 5, figsize=(15, 10))
for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        img = predimage_sample[i].numpy()
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')


def param_count(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print('Total number of parameters : ', sum(params))


class CNN(nn.Module):
    def __init__(self, input_shape, output):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=32,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 4 * 4
                      , out_features=output)
        )

    def forward(self, x):
        x = self.block1(x)
        #         print(f"The output shape of conv block1 is : {x.shape}\n\n")
        x = self.block2(x)
        #         print(f"The output shape of conv block2 is : {x.shape}\n\n")
        x = self.fully_connected_layer(x)
        return x


torch.manual_seed(42)
model=CNN(
    input_shape = 3 ,
    output=len(classes)
)

in_shape = (3, 250, 250)

criterion=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),lr=0.001)

epochs = 10
training_acc = []
training_loss = []

for i in tqdm(range(epochs)):
    epoch_loss = 0
    epoch_acc = 0

    for batch, (x_train, y_train) in enumerate(train_dataloader):
        y_pred = model.forward(x_train)

        loss = criterion(y_pred, y_train)

        if batch % 100 == 0:
            print(f"Looked at {batch * len(x_train)}/{len(train_dataloader.dataset)} samples.")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss
        epoch_acc += accuracy_score(y_train, y_pred.argmax(dim=1))

    training_loss.append((epoch_loss / len(train_dataloader)).detach().numpy())
    training_acc.append(epoch_acc / len(train_dataloader))

    print(
        f"Epoch {i}: Accuracy: {(epoch_acc / len(train_dataloader)) * 100}, Loss: {(epoch_loss / len(train_dataloader))}\n\n")


plt.subplots(figsize=(6,4))
plt.plot(range(epochs),training_loss,color="blue",label="Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.subplots(figsize=(6,4))
plt.plot(range(epochs),training_acc,color="green",label="Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss = 0
test_acc = 0
test_preds = []
test_targets = []
torch.manual_seed(42)
with torch.no_grad():
    for x_test, y_test in test_dataloader:
        y_pred = model.forward(x_test)
        test_pred = torch.softmax(y_pred, dim=1).argmax(dim=1)
        test_preds.append(test_pred)
        test_targets.extend(y_test)

        loss = criterion(y_pred, y_test)
        test_loss += loss
        test_acc += accuracy_score(y_test, y_pred.argmax(dim=1))

test_loss /= len(test_dataloader)
test_acc /= len(test_dataloader)
test_preds = torch.cat(test_preds)
test_targets = torch.Tensor(test_targets)

print(f"The loss of the testing set is : {test_loss}\n")
print(f"The accuracy of the testing set is : {(test_acc*100):0.2f}%\n")