import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    median = cv2.medianBlur(frame,5)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # define range of green color in HSV
    lower_green = np.array([50,50,50])
    upper_green = np.array([70,255,255])

    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = blue_mask + green_mask

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(median,median, mask = mask)

    cv2.imshow('blur',median)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
