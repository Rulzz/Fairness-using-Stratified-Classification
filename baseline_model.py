from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from helper import split, populate_result, get_gender, populate_output, feature_selection
import numpy as np
from collections import Counter


def baseline_model(dataset, model_name, is_feature_sel_req, final_output, interim_output):

    result = {
        "true_total": [],
        "pred_total": [],
        "true_males": [],
        "pred_males": [],
        "true_females": [],
        "pred_females": []
    }

    X, y = split(dataset)
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if is_feature_sel_req:
            X_train = feature_selection(X_train, y_train)
            X_test = feature_selection(X_test, y_test)
        output = []
        for i in range(11):
            if is_feature_sel_req:
                X_train = feature_selection(X_train, y_train)
                X_test = feature_selection(X_test, y_test)
            classifier = DecisionTreeClassifier(criterion='entropy')
            classifier.fit(X_train, y_train)

            output.append(classifier.predict(X_test)[0])
        # Majority Voting
        output_votes = Counter(np.asarray(output))
        predicted, votes = output_votes.most_common()[0]


        populate_result(result, y_test[0], predicted, get_gender(dataset, test_index))
    # print_model_result(result, model_name, final_output)
    populate_output(result, final_output, model_name, interim_output)