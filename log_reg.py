# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imutils import paths
from svm import extract_color_histogram
from numpy import ravel
import numpy as np
import argparse
import imutils
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the data matrix and labels list
data = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract a color histogram from the image, then update the
    # data matrix and labels list
    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)

    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(data), labels, test_size=0.25, random_state=42)

classifier = LogisticRegression()
classifier.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = classifier.predict(testData)
ll = log_loss(testLabels, predictions, eps=1e-15, normalize=True, sample_weight=None, labels=[0, 1])
print("log loss", ll)

print(classification_report(testLabels, predictions, target_names=le.classes_))

tn, fp, fn, tp = confusion_matrix(testLabels, predictions).ravel()
print(tn, fp, fn, tp)
