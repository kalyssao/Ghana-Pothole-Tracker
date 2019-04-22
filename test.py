from joblib import dump, load

import os
import cv2
import imutils

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

svmFit = load('svmFit.joblib')
data = []
labels = []
# load the test images
folderPath = "/Users/lvz/PycharmProjects/pothole_classifier/test"
for imageName in os.listdir(folderPath):
    if imageName != ".DS_Store":
        data = []
        print("[INFO] classifying {}".format(imageName))
        image = cv2.imread(folderPath + "/" + imageName)

        # extract a color histogram feature from the image
        feature = extract_color_histogram(image)
        data.append(feature)

        label = svmFit.predict(data)
        labels.append(label)
        label = "{}".format(label)
        # draw the class and probability on the test image and display it
        #  to our screen
        print(labels)
        cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.imwrite("result_" + imageName, image)

    else:
        continue
