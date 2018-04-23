import cv2
import numpy as np
import matplotlib

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # define range of blue color in HSV
    lower_blue = np.array([0,0,80])
    upper_blue = np.array([0,0,100])

    # define range of green color in HSV
    lower_green = np.array([0,80,0])
    upper_green = np.array([0,100,0])

    # Threshold the HSV image to get only blue and green colors
    blue_mask = cv2.inRange(rgb, lower_blue, upper_blue)
    green_mask = cv2.inRange(rgb, lower_green, upper_green)

    mask = blue_mask + green_mask

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask = mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
