from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
from utils import load_images_to_train
from feature import calculate_features
import constants

print("Running forest")


def persist_random_forest(clf, path):
    joblib.dump(clf, path)


def load_random_forest(path):
    return joblib.load(path)


def train():
    print("Start loading images")
    data_set = load_images_to_train()
    x = data_set["x"]
    y = data_set["y"]
    print("Start training random forest...")
    clf = RandomForestClassifier(random_state=0, bootstrap=True)
    clf.fit(x, y)
    print("Finished training. Persisting trained forest")
    persist_random_forest(clf, constants.path_random_forest)
    print("Persisting done")


def classify(path, path_normal):
    frame = calculate_features(path)
    anger_normal = calculate_features(path_normal)
    v = np.subtract(frame, anger_normal)
    clf = load_random_forest(constants.path_random_forest)
    prediction_prob = clf.predict_proba([v])
    max_index = np.argmax(prediction_prob)
    max_class = clf.classes_[max_index]
    max_prob = prediction_prob[0][max_index]
    print("prediction_prob forest: ", max_class)
    return {
        "class": max_class,
        "prob": max_prob
    }
