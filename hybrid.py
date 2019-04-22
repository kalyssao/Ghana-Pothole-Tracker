#    Title: linear_classifier.py
#    Author: A. Rosebrock
#    Date: August 22, 2016
#    Code version: 1
#    Availability: https://www.pyimagesearch.com/2016/08/22/an-intro-to-linear-classification-with-python/

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import load
import numpy as np
from imutils import paths
import argparse
import os

# importing inceptionV3 model
model = InceptionV3(weights='imagenet', include_top=False)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the feature list and labels list
labels = []
feature_list = []


# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image (assuming that our path has the format: /path/to/dataset/{class}.{image_num}.jpg
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    image = load_img(imagePath, target_size=(299, 299))
    img_data = img_to_array(image)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    feature = model.predict(img_data)
    feature_np = np.array(feature)
    feature_list.append(feature_np.flatten())
    labels.append(label)
    if i > 0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

feature_list_np = np.array(feature_list)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(feature_list_np, labels, test_size=0.25, random_state=42)

# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
#tuned_svm = load('tuned_svmFit.joblib')
tuned_svm = LinearSVC(C=100).fit(trainData, trainLabels)
# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = tuned_svm.predict(testData)

print("[INFO] classification report...")
#log_loss(testLabels, predictions, eps=1e-15, normalize=True, sample_weight=None, labels=[0, 1])
print(classification_report(testLabels, predictions, target_names=le.classes_))

print("[INFO] confusion matrix (tp, fp, tn, tp)...")
tn, fp, fn, tp = confusion_matrix(testLabels, predictions).ravel()
print(tn, fp, fn, tp)
