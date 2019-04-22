# Kalyssa A. Owusu
# Final Capstone Project Code

# Importing necessary modules and functions
import os
import cv2
import time
import serial
import string
import MySQLdb
import pynmea2

from picamera.array import PiRGBArray
from picamera import PiCamera
from joblib import dump, load

# initialising the ports and serial for the NMEA GPS sensor
port = "/dev/ttyS0"
ser = serial.Serial(port, baudrate=9600, timeout=0.5)

# database connection initialisation
db = MySQLdb.connect(host="localhost", user="root", passwd="password", db="locations_data")
cur = db.cursor()

# initialises the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allows camera warmup
time.sleep(0.1)

def readLocation():


while 1:
    # counter to keep track of number of positives
    numPotholes = 0
    threshold = 10
    # keep track of GPS
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
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array

    # For frame in video, extract features and predict
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        # can do resizing in here
        image = frame.array

        imgVec = extract_color_histogram(image)
        svmFit = load('svmFit.joblib')
        res = svmFit.predict(imgVec)

        if res == 1:
            # increment potholes, check if above threshold
            # if so, pass the number of potholes as well (?)
            numPotholes += 1

            if numPotholes >= threshold:
                transmit(lat, lng)

        else:
            numPotholes = 0
