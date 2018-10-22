import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import statistics

csv_path = "/dataset.csv"
# Importing the dataset
dataset = pd.read_csv(csv_path)


def get_male_female_idx(test_idx):
    male_indices = []
    female_indices = []
    counter = 0
    for idx in test_idx:
        if dataset.iloc[idx]['gender'] == 1:
            male_indices.append(counter)
        else:
            female_indices.append(counter)
        counter = counter + 1
    return male_indices, female_indices

features = [0,3,5,8,10]
X = dataset.iloc[:, features].values
y = dataset.iloc[:, -1].values


avg_acc=[]
avg_male_acc=[]
avg_female_acc=[]
for i in range(1):
    kf = KFold(n_splits=10, shuffle=True, random_state=3)
    # kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    acc = []
    male_acc = []
    female_acc = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        male_idx, female_idx = get_male_female_idx(test_index)

        classifier = RandomForestClassifier(criterion='entropy')
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, predicted) * 100
        male_accuracy = accuracy_score(y_test[male_idx], predicted[male_idx]) * 100
        female_accuracy = accuracy_score(y_test[female_idx], predicted[female_idx]) * 100
        acc.append(accuracy)
        if len(male_idx) != 0:
            male_acc.append(male_accuracy)
        if len(female_idx) != 0:
            female_acc.append(female_accuracy)
    avg_acc.append(statistics.mean(acc))
    avg_male_acc.append(statistics.mean(male_acc))
    avg_female_acc.append(statistics.mean(female_acc))

print('###############')
print('Total avg accuracy - ', statistics.mean(avg_acc))
print('Total avg MALE accuracy - ', statistics.mean(avg_male_acc))
print('Total avg FEMALE accuracy - ', statistics.mean(avg_female_acc))
