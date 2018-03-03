from random_forest import classify, train
import constants
from utils import read_all_lines, prefix_string_each, read_all_files


def analyze(source_path, test_class):
    all_lines = read_all_lines(source_path)
    paths = prefix_string_each(constants.DB_PATH, all_lines)
    paths = paths[int(len(paths)*0.7):len(paths)]  # Get only 30% of data for testing
    correct_count = 0
    for path in paths:
        files = read_all_files(path, '*.png')
        neutral = files[0]
        apex = files[len(files)-1]
        classification = classify(apex, neutral)
        classification_class = classification["class"]
        print(neutral)
        print(apex)
        print(classification_class)
        print('-------------------')
        if classification_class == test_class:
            correct_count += 1
    print('Total count: ', len(paths))
    print('Correct count for ' + test_class + ' is: ' + str(correct_count))
    print('% correct: '+str(correct_count/len(paths)))


# Remove below comment line for test
# analyze(constants.ANGER_PATH, constants.ANGER)
# analyze(constants.HAPPY_PATH, constants.HAPPY)
# analyze(constants.SADNESS_PATH, constants.SADNESS)
analyze(constants.SURPRISE_PATH, constants.SURPRISE)


# Remove comment for below line to start training
# train()
