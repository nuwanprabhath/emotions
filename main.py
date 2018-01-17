import numpy as np
import cv2
from skimage import feature


def flip_frame(frame_to_flip):
    return cv2.flip(frame_to_flip, 1)


#
# capture = cv2.VideoCapture(0)
# numPoints = 24
# radius = 8
# while True:
#     if capture.isOpened():
#         ret, frame = capture.read()
#         if ret:
#             flippedFrame = flip_frame(frame)
#             gray_scale_frame = cv2.cvtColor(flippedFrame, cv2.COLOR_BGR2GRAY)
#             lbp = feature.local_binary_pattern(gray_scale_frame, numPoints, radius, method="uniform")
#             cv2.imshow('Capture', lbp)
#             cv2.imshow('Capture1', gray_scale_frame)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 break


from imutils.video import VideoStream
from imutils import face_utils
import datetime
import imutils
import time
import dlib


def get_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# Download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# vs = VideoStream().start()
# time.sleep(2.0)
pointsConsider = [38, 39, 41, 42, 44, 48, 45, 47, 20, 40, 21, 22, 19, 23, 43, 24, 25, 26, 49, 55, 52, 58, 34, 50, 54]
leftEye = [(38, 42), (39, 41)]
rightEye = [(44, 48), (45, 47)]
leftEyebrow = [(20, 40), (21, 40), (22, 40), (19, 40)]
rightEyebrow = [(23, 43), (24, 43), (25, 43), (26, 43)]
mouthWidth = [(49, 55)]
mouthHeight = [(52, 58)]
angleWithNoseLeft = [(34, 49), (34, 55)]
angleWithNoseRight = [(34, 50), (34, 54)]
verticalWithNose = [(34, 52)]
pairs = leftEye + rightEye + leftEyebrow + rightEyebrow + mouthWidth + mouthHeight + angleWithNoseLeft + \
        angleWithNoseRight + verticalWithNose

eyeDistance = 0
eyebrowDistanceLeft = 0
eyebrowDistanceRight = 0
mouthWidthDistance = 0
mouthHeightDistance = 0
angleWithNoseLeftDistance = 0
angleWithNoseRightDistance = 0
distances = {}


# while True:
#     # grab the frame from the threaded video stream, resize it to
#     # have a maximum width of 400 pixels, and convert it to
#     # grayscale
#     frame = vs.read()
#     frame = imutils.resize(frame)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     landmarks = {}
#     # detect faces in the grayscale frame
#     rects = detector(gray, 0)
#     # loop over the face detections
#     for rect in rects:
#         # print("new detect")
#         # determine the facial landmarks for the face region, then
#         # convert the facial landmark (x, y)-coordinates to a NumPy
#         # array
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         # loop over the (x, y)-coordinates for the facial landmarks
#         # and draw them on the image
#         c = 0
#         for (x, y) in shape:
#             c += 1
#             # print('landmarks:', shape)
#             # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
#             if c in pointsConsider:
#                 landmarks[c] = (x, y)
#             cv2.putText(frame, str(c), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)
#
#     currentEyebrowLeft = 0
#     currentEyebrowRight = 0
#     currentAngleWithNoseLeft = 0
#     currentAngleWithNoseRight = 0
#     for pair in pairs:
#         p1 = pair[0]
#         p2 = pair[1]
#         try:
#             l1 = landmarks[p1]
#             l2 = landmarks[p2]
#             distance = get_distance(l1, l2)
#             lipNormalDistance = get_distance(landmarks[34], landmarks[52])
#             eyeNormDistanceLeft = get_distance(landmarks[40], landmarks[22])
#             eyeNormDistanceRight = get_distance(landmarks[43], landmarks[23])
#
#             if pair in leftEyebrow:
#                 currentEyebrowLeft += distance / eyeNormDistanceLeft
#             elif pair in rightEyebrow:
#                 currentEyebrowRight += distance / eyeNormDistanceRight
#             elif pair in mouthWidth:
#                 distances['mouthWidthDistance'] = distance/lipNormalDistance - mouthWidthDistance
#                 mouthWidthDistance = distance/lipNormalDistance
#             elif pair in mouthHeight:
#                 distances['mouthHeightDistance'] = distance/lipNormalDistance - mouthHeightDistance
#                 mouthHeightDistance = distance/lipNormalDistance
#             elif pair in angleWithNoseLeft:
#                 currentAngleWithNoseLeft += distance/lipNormalDistance
#             elif pair in angleWithNoseRight:
#                 currentAngleWithNoseRight += distance/lipNormalDistance
#             cv2.line(frame, l1, l2, (0, 255, 0), 1, cv2.LINE_AA)
#         except KeyError:
#             print('not found pairs1', p1)
#             print('not found pairs2', p2)
#
#     distances['eyebrowDistanceLeft'] = currentEyebrowLeft - eyebrowDistanceLeft
#     distances['eyebrowDistanceRight'] = currentEyebrowRight - eyebrowDistanceRight
#     distances['angleWithNoseLeftDistance'] = currentAngleWithNoseLeft - angleWithNoseLeftDistance
#     distances['angleWithNoseRightDistance'] = currentAngleWithNoseRight - angleWithNoseRightDistance
#     eyebrowDistanceLeft = currentEyebrowLeft
#     eyebrowDistanceRight = currentEyebrowRight
#     angleWithNoseLeftDistance = currentAngleWithNoseLeft
#     angleWithNoseRightDistance = currentAngleWithNoseRight
#
#     print('Distances:', distances['eyebrowDistanceLeft'])
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


def calculate_features(path, out_path=''):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    # print(frame)
    # frame = imutils.resize(frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = {}
    # detect faces in the grayscale frame
    rects = detector(frame, 0)
    # loop over the face detections
    for rect in rects:
        # print("new detect")
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        c = 0
        for (x, y) in shape:
            c += 1
            if c in pointsConsider:
                landmarks[c] = (x, y)
            cv2.putText(frame, str(c), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1, cv2.LINE_AA)

    current_eyebrow_left = 0
    current_eyebrow_right = 0
    current_angle_with_nose_left = 0
    current_angle_with_nose_right = 0
    mouth_width_distance = 0
    mouth_height_distance = 0

    for pair in pairs:
        p1 = pair[0]
        p2 = pair[1]
        try:
            l1 = landmarks[p1]
            l2 = landmarks[p2]
            distance = get_distance(l1, l2)
            lip_normal_distance = get_distance(landmarks[34], landmarks[52])
            eye_norm_distance_left = get_distance(landmarks[40], landmarks[22])
            eye_norm_distance_right = get_distance(landmarks[43], landmarks[23])

            if pair in leftEyebrow:
                current_eyebrow_left += distance / eye_norm_distance_left
            elif pair in rightEyebrow:
                current_eyebrow_right += distance / eye_norm_distance_right
            elif pair in mouthWidth:
                distances['mouthWidthDistance'] = distance / lip_normal_distance - mouth_width_distance
                mouth_width_distance = distance / lip_normal_distance
            elif pair in mouthHeight:
                distances['mouthHeightDistance'] = distance / lip_normal_distance - mouth_height_distance
                mouth_height_distance = distance / lip_normal_distance
            elif pair in angleWithNoseLeft:
                current_angle_with_nose_left += distance / lip_normal_distance
            elif pair in angleWithNoseRight:
                current_angle_with_nose_right += distance / lip_normal_distance
            cv2.line(frame, l1, l2, (0, 255, 0), 1, cv2.LINE_AA)
        except KeyError:
            print('not found pairs1', p1)
            print('not found pairs2', p2)

    out = np.array([current_eyebrow_left, current_eyebrow_right, current_angle_with_nose_left,
                    current_angle_with_nose_right, mouth_width_distance, mouth_height_distance])
    # print('out:', out)
    if len(out_path) > 0:
        cv2.imwrite(out_path, frame)
    return out

# calculate_features('data/1_anger/s5/1.png', 'data/output/1-1.png')
