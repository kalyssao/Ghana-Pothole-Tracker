#    Camera setup code adapted from test_image.py by A. Rosebrock
#    Availability: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/

# import the necessary packages
import imutils
import cv2
import time
import serial
import string
import MySQLdb
import pynmea2
from joblib import load


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


def transmit(latitude, longitude):
    # initialising the database connection
    username = "admin"
    db = MySQLdb.connect(host="localhost", user="root", passwd="password", db="fix_ghanas_potholes")
    cur = db.cursor()
    sql = "INSERT INTO location_data (username, latitude, longitude) VALUES (%s, %f,%f)"

    try:
        print("Inserting into database")
        cur.execute(sql, username, latitude, longitude)
        db.commit()
        print("done!")

    except:
        db.rollback()
        print("write failed :(")

    cur.close()
    db.close()


# main function contains while loop
def main():

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
        found = False

        # initialising the ports and serial for the NMEA strings
        with serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1) as ser:
            newData = ser.readline().decode('ascii', errors='replace')

            # doesn't grab frame until the lat and lng are found
            if newData[0:6] == '$GPGGA':
                newMsg = pynmea2.parse(newData)
                lat = newMsg.latitude
                lng = newMsg.longitude
                found = True
            else:
                found = False
                continue

        # Monitor video feed, and apply model on each image (time delay?)
        camera.capture(raw_capture, format="bgr")
        image = raw_capture.array

        # For frame in video, extract features and predict
        for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            # can do resizing in here
            image = frame.array

            img_vec = extract_color_histogram(image)
            svmFit = load('svmFit.joblib')
            res = svmFit.predict(img_vec)   # returns a vector

            if 1 in res:
                # increment potholes, check if above threshold
                # if so, pass the number of potholes as well (?)
                num_potholes += 1

                if num_potholes >= threshold and found:
                    transmit(lat, lng)

            else:
                num_potholes = 0


if __name__ == "main":
    main()
