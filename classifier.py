from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np
import cv2
import glob

from main import calculate_features

path_neural_net = 'data/emotions_net.pkl'


def persist_neural_net(net, path):
    joblib.dump(net, path)


def load_neural_net(path):
    return joblib.load(path)


def load_images():
    x = []
    y = []
    print("Start loading images...")

    anger_normal = calculate_features('data/1_anger/s5/n.png', 'output/anger/n.png')
    c = 0
    for path in glob.glob("data/1_anger/s5/*.png"):
        if not ('n.png' in path):
            c += 1
            v = calculate_features(path, 'data/output/anger/' + str(c) + '.png')
            x.append(np.subtract(v, anger_normal))
            y.append('anger')

    happy_normal = calculate_features('data/5_happy/s10/n.png', 'output/happy/n.png')
    c = 0
    for path in glob.glob("data/5_happy/s10/*.png"):
        if not ('n.png' in path):
            c += 1
            v = calculate_features(path, 'data/output/happy/' + str(c) + '.png')
            x.append(np.subtract(v, happy_normal))
            y.append('happy')

    sadness_normal = calculate_features('data/6_sadness/s14/n.png', 'output/sadness/n.png')
    c = 0
    for path in glob.glob("data/6_sadness/s14/*.png"):
        if not ('n.png' in path):
            c += 1
            v = calculate_features(path, 'data/output/sadness/' + str(c) + '.png')
            x.append(np.subtract(v, sadness_normal))
            y.append('sadness')

    surprise_normal = calculate_features('data/7_surprise/s10/n.png', 'output/surprise/n.png')
    c = 0
    for path in glob.glob("data/7_surprise/s10/*.png"):
        if not ('n.png' in path):
            c += 1
            v = calculate_features(path, 'data/output/surprise/' + str(c) + '.png')
            x.append(np.subtract(v, surprise_normal))
            y.append('surprise')

    data = {"x": x, "y": y}
    print("Loading images finished", data)
    return data


def train():
    data_set = load_images()
    x = data_set["x"]
    y = data_set["y"]
    print("Start training neural network...")
    # Classification http://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification
    # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    clf = MLPClassifier(verbose=True, solver='adam', alpha=0.0001, hidden_layer_sizes=(10, 10, 10), random_state=1,
                        early_stopping=False, max_iter=500)
    clf.fit(x, y)
    print("Finished training. Persisting trained neural net")
    persist_neural_net(clf, path_neural_net)
    print("Persisting done")


def classify(path, path_normal):
    # test_image1 = read_and_transform_image('data/train/1/1-2.png')
    # test_image2 = read_and_transform_image('data/train/2/2-3.png')
    # prediction = clf.predict([test_image1])
    frame = calculate_features(path, 'data/output/anger/1.png')
    anger_normal = calculate_features(path_normal, 'data/output/anger/n.png')
    v = np.subtract(frame, anger_normal)

    clf = load_neural_net(path_neural_net)
    prediction_prob = clf.predict_proba([v])
    max_index = np.argmax(prediction_prob)
    max_class = clf.classes_[max_index]
    max_prob = prediction_prob[0][max_index]
    print("prediction_prob: ", max_class)
    return {
            "class": max_class,
            "prob": max_prob
        }


# train()
# classify('data/1_anger/s5/1.png', 'data/1_anger/s5/n.png')
# classify('data/test/surprise/s34/4.png', 'data/test/surprise/s34/n.png')
# classify('data/test/happy/s50/3.png', 'data/test/happy/s50/n.png')
classify('data/test/anger/s37/4.png', 'data/test/anger/s37/n.png') #incorrect
# classify('data/5_happy/s10/S010_006_00000010.png', 'data/5_happy/s10/n.png')
