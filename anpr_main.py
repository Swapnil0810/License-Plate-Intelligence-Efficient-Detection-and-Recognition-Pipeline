from sort import SORT
from cv2 import imshow, waitKey, rectangle, putText, FONT_HERSHEY_DUPLEX, LINE_AA, polylines, fillPoly, bitwise_and, VideoCapture
import numpy as np
import os
import time
from ultralytics import YOLO
from service import number_plate_detection


file = "Put the RTSP of your CCTV"

cap = cv2.VideoCapture(file)
temp = {}
mot_tracker = Sort()

while True:
    ret,frame = cap.read()
    track, np = number_plate_detection(frame, temp, mot_tracker)

    x1, y1, x2, y2, id = track
    frame = rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 3)
    frame = putText(frame, np, (int(x1), int(y1)), FONT_HERSHEY_DUPLEX, 5, (0, 255, 0), 2)
    imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()