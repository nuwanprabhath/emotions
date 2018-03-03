import numpy as np
import cv2
from imutils import face_utils
import dlib


def flip_frame(frame_to_flip):
    return cv2.flip(frame_to_flip, 1)


def get_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("loading facial landmark predictor...")
face_detector = dlib.get_frontal_face_detector()
# Download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
landmark_shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
pointsConsider = [38, 39, 41, 42, 44, 48, 45, 47, 20, 40, 21, 22, 19, 23, 43, 24, 25, 26, 49, 55, 52, 58, 34, 50, 51, 53, 54]
leftEye = [(38, 42), (39, 41)]
rightEye = [(44, 48), (45, 47)]
leftEyebrow = [(20, 40), (21, 40), (22, 40), (19, 40)]
rightEyebrow = [(23, 43), (24, 43), (25, 43), (26, 43)]
mouthWidth = [(49, 55)]
mouthHeight = [(52, 58)]
angleWithNoseLeft = [(34, 49), (34, 50), (34, 51)]
angleWithNoseRight = [(34, 53), (34, 54), (34, 55)]
verticalWithNose = [(34, 52)]
pairs = leftEyebrow + rightEyebrow + mouthWidth + mouthHeight + angleWithNoseLeft + \
        angleWithNoseRight + verticalWithNose

eyeDistance = 0
eyebrowDistanceLeft = 0
eyebrowDistanceRight = 0
mouthWidthDistance = 0
mouthHeightDistance = 0
angleWithNoseLeftDistance = 0
angleWithNoseRightDistance = 0
distances = {}


def calculate_features(path, out_path=''):
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    landmarks = {}
    # detect faces in the grayscale frame
    rects = face_detector(frame, 0)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = landmark_shape_predictor(frame, rect)
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
                mouth_width_distance = distance / lip_normal_distance
            elif pair in mouthHeight:
                mouth_height_distance = distance / lip_normal_distance
            elif pair in angleWithNoseLeft:
                current_angle_with_nose_left += distance / lip_normal_distance
            elif pair in angleWithNoseRight:
                current_angle_with_nose_right += distance / lip_normal_distance
            cv2.line(frame, l1, l2, (0, 255, 0), 1, cv2.LINE_AA)
        except KeyError:
            print('not found pairs 1', p1)
            print('not found pairs 2', p2)

    out = np.array([current_eyebrow_left, current_eyebrow_right, current_angle_with_nose_left,
                    current_angle_with_nose_right, mouth_width_distance, mouth_height_distance])
    # print('out:', out)
    if len(out_path) > 0:
        cv2.imwrite(out_path, frame)
    return out
