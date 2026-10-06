import cv2
from matplotlib import pyplot as plt
vid_path = 'basketball.mp4'
ann_file = 'annotation.txt'


cap = cv2.VideoCapture(vid_path)
ret, frame = cap.read()

with open(ann_file, 'r') as f:
    first_line = f.readline().strip()
    x_min, y_min, x_max, y_max = map(int, first_line.split(','))

cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

cv2.imshow('1st frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()