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

#
# from imutils.video import VideoStream
# from imutils import face_utils
# import datetime
# import imutils
# import time
# import dlib
#
# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor
# print("loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# # Download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# print("camera sensor warming up...")
# vs = VideoStream().start()
# # time.sleep(2.0)
#
# while True:
#     # grab the frame from the threaded video stream, resize it to
#     # have a maximum width of 400 pixels, and convert it to
#     # grayscale
#     frame = vs.read()
#     frame = imutils.resize(frame)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # detect faces in the grayscale frame
#     rects = detector(gray, 0)
#     # loop over the face detections
#     for rect in rects:
#         print("new detect")
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         print(rect)
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         print(shape.size == 136)
#         # loop over the (x, y)-coordinates for the facial landmarks
#         # and draw them on the image
#         for (x, y) in shape:
#             # print('landmarks:', shape)
#             cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
#
#     # show the frame
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break
#
# cv2.destroyAllWindows()
# vs.stop()
