from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np
import cv2

path_neural_net = 'data/emotions_net.pkl'

def load_images():
    x = []
    y = []
    print("Start loading images...")

    data = {"x": x, "y": y}
    print("Loading images finished")
    return data

