# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 16:54:00 2020

@author: ARSHPREET SINGH
"""

# If you don’t have imutils  in your virtual environment, you can install it via:

"""
$ pip install imutils

"""

# Run the following command in cmd (Windows) in order to execute this program.
# Make sure to change the default directory to working directory before executing.
# Also, make sure you have already set the virtual environment for Python.

"""
$ python detect_faces_video.py --prototxt deploy.prototxt.txt \
$ --model res10_300x300_ssd_iter_140000.caffemodel

"""

# Using the argparse Python package, we can easily parse 
# command line arguments in the terminal/command line.

from imutils.video import VideoStream
import imutils
import time
import numpy as np
import cv2
import argparse

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help= "path to Caffe (deploy) prototxt file")
ap.add_argument("-m", "--model", required = True, help= "path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type= float, default= 0.5, help= "minimum probability to filter weak detections")
# ap.add_argument("-o", "--output", required = True, help= "output image")
args = vars(ap.parse_args())

# Load model from disk
print("[INFO] Loading Model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])



# Initialize the video stream from your laptop's built-in camera
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# Here, we initialize a VideoStream object specifying camera with index 0
# as the source (in general this would be our laptop’s built in camera or our
# desktop’s first camera detected).
# If you want to parse a video file (instead of a video stream), 
# use FileVideoStream instead of VideoStream above.



# Loop over the frames from the video stream and compute face detections using OpenCV...
while True:
    # Grab the frames from the video stream thread and
    # resize it to have a maximum width of 400 pixels.
    frames = vs.read()
    frames = imutils.resize(frames, width=400)

    # Grab the frame's dimensions and convert it into a blob
    (h, w) = frames.shape[:2]
    blob= cv2.dnn.blobFromImage(cv2.resize(frames, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
  
        # In case the our text  would go off-image (such as when the 
        # face detection occurs at the very top of an image), we shift
        # it down by 10 pixels.
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frames, (startX, startY), (endX, endY),(0, 0, 255), 4)
        cv2.putText(frames, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frames)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

# Press "q" to exit the program
