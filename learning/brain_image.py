import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


for dirname, _, filenames in os.walk('/Users/andrey/Downloads/brain_mri_scan_images'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


positive_folder = "/Users/andrey/Downloads/brain_mri_scan_images/positive"  # Path for positive folder

negative_folder = "/Users/andrey/Downloads/brain_mri_scan_images/negative"      # Path for negative folder


def load_and_display_image(folder, label):
    img_name = os.listdir(folder)[0]  # Assuming there's at least one image in each folder
    img_path = os.path.join(folder, img_name)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR, convert to RGB for display

    plt.imshow(img)
    plt.title(f"{label} Image")
    plt.axis('off')
    plt.show()


# Load and display a sample positive image
load_and_display_image(positive_folder, "Positive")

# Load and display a sample negative image
load_and_display_image(negative_folder, "Negative")


def preprocess_images(folder, label, image_size=(224, 224)):
    data = []
    labels = []

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize pixel values to the range [0, 1]

        data.append(img)
        labels.append(label)

    return np.array(data), np.array(labels)


# Preprocess positive images
positive_data, positive_labels = preprocess_images(positive_folder, 1)

# Preprocess negative images
negative_data, negative_labels = preprocess_images(negative_folder, 0)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add your own classification layers on top
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the transfer learning model
model_transfer = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model_transfer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
