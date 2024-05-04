import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
