import glob
from feature import calculate_features
import numpy as np
import constants


def read_all_lines(path):
    lines = [line.rstrip('\n') for line in open(path)]
    return lines


def prefix_string_each(string, string_list):
    lines = [string + line for line in string_list]
    return lines


def read_all_files(path, extension_pattern):
    all_files = glob.glob(path + '/' + extension_pattern)
    all_files.sort()
    return all_files


def load_feature_and_label(x, y, emotion, emotion_path):
    all_lines = read_all_lines(emotion_path)
    paths = prefix_string_each(constants.DB_PATH, all_lines)
    print(emotion + " training count: ", int(len(paths) * 0.7))
    print(emotion + " remaining count: ", len(paths) - int(len(paths) * 0.7))
    paths = paths[0:int(len(paths) * 0.7)]
    for path in paths:
        files = read_all_files(path, '*.png')
        neutral = files[0]
        apex = files[len(files) - 1]
        normal_feature = calculate_features(neutral)
        apex_feature = calculate_features(apex)
        x.append(np.subtract(apex_feature, normal_feature))
        y.append(emotion)


def load_images_to_train():
    x = []
    y = []

    load_feature_and_label(x, y, constants.ANGER, constants.ANGER_PATH)
    load_feature_and_label(x, y, constants.HAPPY, constants.HAPPY_PATH)
    load_feature_and_label(x, y, constants.SADNESS, constants.SADNESS_PATH)
    load_feature_and_label(x, y, constants.SURPRISE, constants.SURPRISE_PATH)

    data = {"x": x, "y": y}
    print("Loading images finished in new load", data)
    return data
