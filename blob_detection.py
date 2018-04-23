#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;
import math

map = np.zeros((6, 12))

pixel_x = 960
pixel_y = 540
width = 80
height = 90

# Read image
im = cv2.imread("blue_circles2.png", cv2.IMREAD_COLOR)
# Get camera input
#cap = cv2.VideoCapture(0)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 2000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.9

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.12

# Filter by Color
#params.filterByColor = True
#params.blobColor = 100

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

#Print out initial array, should be all 0s
#for i in range(0, n):
#        for j in range(0, n):
#                print map[i][j]

# Detect blobs.
keypoints = detector.detect(im)
##print keypoints
##
##for keypoint in keypoints:
##        x = keypoint.pt[0]
##        y = keypoint.pt[1]
##        print x
##        print y
##        print ("The last is the great fairy doodles")
##        i = int(x/width)
##        j = int(y/height)
##        print i
##        print j
##        print ("kjsldkjflajlf")
##        map[j][i] = 1
##        for a in range(0, 6):
##                for b in range(0, 12):
##                        print map[a][b]
##                print("blurpleberry")

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
