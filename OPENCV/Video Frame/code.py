# Detect the objects

import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\aisha\AVSCODE\OPENCV\Video Frame\los_angeles.mp4")

ret, frame = cap.read()

if ret:
    cv2.imshow("Frame", frame)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()


#-----------------------------------

# video is nothing 1 image after another image
# if i press key then it keep goes to the next frame.

import cv2
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\aisha\AVSCODE\OPENCV\Video Frame\los_angeles.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()


#--------------------------------------

# it will display the complete video

import cv2
import numpy as np

cap = cv2.VideoCapture(r'C:\Users\aisha\AVSCODE\OPENCV\Video Frame\los_angeles.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()