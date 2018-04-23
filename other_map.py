import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

#Initiate the camera
cap = cv2.VideoCapture(0)

#create subplot
ax = plt.subplot(1,2,1)

#create image plot
im = ax.imshow(grab_frame(cap))

def update(i):
    im.set_data(grab_frame(cap))

ani = FuncAnimation(plt.gcf(), update, interval=200)
plt.show()
