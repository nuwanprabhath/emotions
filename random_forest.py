from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
from classifier import load_images
from main import calculate_features


path_random_forest = 'data/emotions_forest.pkl'


def persist_random_forest(clf, path):
    joblib.dump(clf, path)


def load_random_forest(path):
    return joblib.load(path)


def train():
    data_set = load_images()
    x = data_set["x"]
    y = data_set["y"]
    print("Start training random forest...")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(x, y)
    print("Finished training. Persisting trained forest")
    persist_random_forest(clf, path_random_forest)
    print("Persisting done")


def classify(path, path_normal):
    frame = calculate_features(path, 'data/output/anger/1.png')
    anger_normal = calculate_features(path_normal, 'data/output/anger/n.png')
    v = np.subtract(frame, anger_normal)

    clf = load_random_forest(path_random_forest)
    prediction_prob = clf.predict_proba([v])
    max_index = np.argmax(prediction_prob)
    max_class = clf.classes_[max_index]
    max_prob = prediction_prob[0][max_index]
    print("prediction_prob forest: ", max_class)
    return {
        "class": max_class,
        "prob": max_prob
    }


# train()
# classify('data/1_anger/s5/1.png', 'data/1_anger/s5/n.png')
# classify('data/5_happy/s10/S010_006_00000010.png', 'data/5_happy/s10/n.png')
# classify('data/test/anger/s37/4.png', 'data/test/anger/s37/n.png')
classify('data/test/happy/s50/3.png', 'data/test/happy/s50/n.png') #incorrect
# classify('data/test/surprise/s34/4.png', 'data/test/surprise/s34/n.png')
