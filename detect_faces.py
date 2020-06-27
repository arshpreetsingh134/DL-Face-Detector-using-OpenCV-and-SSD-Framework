# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:44:30 2020

@author: ARSHPREET SINGH

"""

# Run the following command in cmd (Windows) in order to execute this program.
# Make sure to change the default directory to working directory before executing.
# Also, make sure you have already set the virtual environment for Python.

"""
$ python detect_faces.py --image test_image.jpg --prototxt deploy.prototxt.txt --model \
$ res10_300x300_ssd_iter_140000.caffemodel --output output_01.jpg

"""

# Using the argparse Python package, we can easily parse 
# command line arguments in the terminal/command line.

import argparse
import cv2
import numpy as np

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help= "path to input image")
ap.add_argument("-p", "--prototxt", required = True, help= "path to Caffe (deploy) prototxt file")
ap.add_argument("-m", "--model", required = True, help= "path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type= float, default= 0.5, help= "minimum probability to filter weak detections")
ap.add_argument("-o", "--output", required = True, help= "output image")
args = vars(ap.parse_args())

# Load model from disk
print("[INFO] Loading Model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load the input image and construct a blob by resizing
# it to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob= cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
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
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 4)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image after execution
#cv2.imshow("output", image)
#cv2.waitKey(0)
cv2.imwrite(args["output"], image)

# After executing, check your directory for the output image "output_01.jpg"

