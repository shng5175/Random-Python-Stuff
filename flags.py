
import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print flags

green = [0,255,0 ]
hsv_green = cv2.cvtColor(green,cv2.COLOR_RGB2HSV)
print hsv_green
