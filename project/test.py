import os
import numpy as np
from imutils import paths
import cv2

imagePaths = list(paths.list_images('Folio'))


# print(len(imagePaths))

def load(imagePaths, width, height):
    # initialize the list of features and labels
    data = []
    labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load image and extract label
        # format: /path/to/dataset/{class}/{image}.jpg
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA)
        label = imagePath.split(os.path.sep)[-2]

        data.append(image)
        labels.append(label)

    return (np.array(data), np.array(labels))


(data, labels) = load(imagePaths, 32, 32)
print(data.shape)
