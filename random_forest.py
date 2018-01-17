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
# anger
# classify('data/test/anger/s37/4.png', 'data/test/anger/s37/n.png') #in
# classify('data/test/anger/s71/4.png', 'data/test/anger/s71/n.png')
# classify('data/test/anger/s87/4.png', 'data/test/anger/s87/n.png')

# happy
# classify('data/test/happy/s50/3.png', 'data/test/happy/s50/n.png')
# classify('data/test/happy/s53/4.png', 'data/test/happy/s53/n.png')
# classify('data/test/happy/s61/4.png', 'data/test/happy/s61/n.png')

# sadness
# classify('data/test/sadness/s42/4.png', 'data/test/sadness/s42/n.png')
# classify('data/test/sadness/s80/4.png', 'data/test/sadness/s80/n.png')
# classify('data/test/sadness/s81/4.png', 'data/test/sadness/s81/n.png') #in

# surprise
# classify('data/test/surprise/s34/4.png', 'data/test/surprise/s34/n.png')
# classify('data/test/surprise/s46/4.png', 'data/test/surprise/s46/n.png')
# classify('data/test/surprise/s63/4.png', 'data/test/surprise/s63/n.png')
