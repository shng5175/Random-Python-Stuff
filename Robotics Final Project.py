import numpy as np
import cv2

n = 4
map = map = np.zeros((n, n))
curr = 0
curc = 0

img = cv2.imread('blue_circles.png',0)

for y in range (50, 400):
    curc = 0
    for x in range (50, 400):
        if np.any(img[x][y]) < 255:
            map[curr][curc] = 1
        curc = curc + 1
        x = x + 100
    curr = curr + 1
    y = y + 100

for i in range (0, 4):
    for j in range(0, 4):
        print map[i][j]
