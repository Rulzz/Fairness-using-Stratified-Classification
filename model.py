from random import *
import random
from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from helper import split, populate_result, get_gender, populate_output, feature_selection


def get_test_set(X, y, test_idx):
    x_test = X[test_idx, :]
    y_test = y[test_idx]
    return [x_test], [y_test]


def get_train_set(X, y, males_idx, females_idx, test_idx):

    if test_idx in males_idx:
        males_idx.remove(test_idx)
    else:
        females_idx.remove(test_idx)

    males_idx = sample(males_idx, 17)       # Random samples of males
    females_idx = sample(females_idx, 17)   # Random samples of females
    train_idx = males_idx + females_idx
    shuffle(train_idx)

    x_train = X[train_idx, :]
    y_train = y[train_idx]

    return x_train, y_train


def model(dataset, model_name, is_feature_sel_req, final_output, interim_output):

    #Dataset size
    dataset_size = 55
    test_selection = np.arange(0, dataset_size, 1)
    random.shuffle(test_selection)
    result = {
        "true_total": [],
        "pred_total": [],
        "true_males": [],
        "pred_males": [],
        "true_females": [],
        "pred_females": []
    }


    # Get male/female indices
    males_idx = dataset.index[dataset['gender'] == 1].tolist()
    females_idx = dataset.index[dataset['gender'] == 2].tolist()

    X, y = split(dataset)

    # if is_feature_sel_req:
    #     X = feature_selection(X, y)

    for iterations in range(dataset_size):
        test_idx = test_selection[0]
        test_selection = np.delete(test_selection, [0])
        x_test, y_test = get_test_set(X, y, test_idx)
        test_gender = get_gender(dataset, test_idx)
        output = []
        # Building Random forest
        for i in range(11):
            X_train, Y_train = get_train_set(X, y, males_idx.copy(), females_idx.copy(), test_idx)
            if is_feature_sel_req:
                X_train = feature_selection(X_train, Y_train)
                x_test = feature_selection(x_test, y_test)
            classifier = DecisionTreeClassifier(criterion='entropy')
            classifier.fit(X_train, Y_train)

            output.append(classifier.predict(x_test)[0])
        # Majority Voting
        output_votes = Counter(np.asarray(output))
        predicted, votes = output_votes.most_common()[0]
        populate_result(result, y_test[0], predicted, test_gender)
    # print_model_result(result, model_name, final_output)
    populate_output(result, final_output, model_name, interim_output)