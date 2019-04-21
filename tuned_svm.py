# USAGE
# python3 svm.py --dataset pothole_classifier

# import the necessary packages
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from imutils import paths
from numpy import ravel
import numpy as np
import argparse
import imutils
import cv2
import os

def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


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

# grid search parameters 
Cs = [0.001, 0.01, 0.1, 1, 100]
param_grid = {'C': Cs}

# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
svmFit = GridSearchCV(estimator=LinearSVC(), param_grid = param_grid, n_jobs=-1)
svmFit.fit(trainData, trainLabels)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = svmFit.predict(testData)
print(classification_report(testLabels, predictions,
    target_names=le.classes_))

tn, fp, fn, tp = confusion_matrix(testLabels, predictions).ravel()
print(tn, fp, fn, tp)

print('Best score:', svmFit.best_score_) 
print('Best C:', svmFit.best_estimator_.C)
print('Best Gamma:', svmFit.best_estimator_.gamma)
