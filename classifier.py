# import the necessary packages
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import time
import serial
import string
import MySQLdb
import pynmea2

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


def SVM_classifier(path):
    # initialize the data matrix and labels list
    data = []
    labels = []
    # grab the list of images that we'll be describing
    folderPath = path
    imagePaths = os.listdir("/Users/lvz/PycharmProjects/pothole_classifier/pothole_classifier")

    # loop over the input images
    for imageName in imagePaths:
        image = cv2.imread(folderPath + "/" + imageName)

        # extract a color histogram feature from the image and update labels and data
        feature = extract_color_histogram(image)
        data.append(feature)

        label = imageName.split(os.path.sep)[-1].split(".")[0]
        labels.append(label)

    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(data), labels, test_size=0.25, random_state=42)

    svmFit = LinearSVC()
    svmFit.fit(trainData, trainLabels)

    return svmFit


def transmit(latitude, longitude):
    # initialising the database connection
    username = "admin"
    db = MySQLdb.connect(host="localhost", user="root", passwd="password", db="fix_ghanas_potholes")
    cur = db.cursor()
    sql = "INSERT INTO location_data (admin, latitude, longitude) VALUES (%f,%f)"

    try:
        print("Inserting into database")
        cur.execute(sql, latitude, longitude)
        db.commit()
        print("done!")

    except:
        db.rollback()
        print("write failed :(")

    cur.close()
    db.close()


# main function contains while loop
def main(model):
    # initialising the ports and serial for the NMEA GPS sensor
    port = "/dev/ttyS0"
    ser = serial.Serial(port, baudrate=9600, timeout=0.5)

    # initialising the camera
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    raw_capture = PiRGBArray(camera, size=(640, 480))

    # allowing the camera to warm up
    time.sleep(0.1)
    while 1:
        # counter to keep track of number of positives
        num_potholes = 0
        threshold = 10

        # keep track of NMEA strings
        dataOut = pynmea2.NMEAStreamReader()
        newData = ser.readline()

        if newData[0:6] == '$GPGGA':
            newMsg = pynmea2.parse(newData)
            lat = newMsg.latitude
            lng = newMsg.longitude
        else:
            # doesn't grab frame until the lat and lng are found
            pass

        # Monitor video feed, and apply model on each image (time delay?)
        camera.capture(raw_capture, format="bgr")
        image = raw_capture.array

        # For frame in video, extract features and predict
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # can do resizing in here
            image = frame.array

            img_vec = extract_color_histogram(image)
            res = model.predict(img_vec)

            if res == 1:
                # increment potholes, check if above threshold
                # if so, pass the number of potholes as well (?)
                num_potholes += 1

                if num_potholes >= threshold & lat & lng:
                    transmit(lat, lng)

            else:
                num_potholes = 0


if __name__ == "main":
    svm = SVM_classifier("/Users/lvz/PycharmProjects/pothole_classifier/pothole_classifier")
    main(svm)
