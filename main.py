import numpy as np
import cv2
from skimage import feature


def flip_frame(frame_to_flip):
    return cv2.flip(frame_to_flip, 1)


capture = cv2.VideoCapture(0)
numPoints = 24
radius = 8
while True:
    if capture.isOpened():
        ret, frame = capture.read()
        if ret:
            flippedFrame = flip_frame(frame)
            gray_scale_frame = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray_scale_frame, numPoints, radius, method="uniform")
            cv2.imshow('Capture', lbp)
            cv2.imshow('Capture1', gray_scale_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
