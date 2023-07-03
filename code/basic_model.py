import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
import utils
from sklearn.tree import DecisionTreeClassifier

model_list = {
    "m1": LogisticRegression(penalty="l1", C=0.5, solver="liblinear"),
    "m2": BernoulliNB(alpha=1.5),
    "m3": SVC(kernel="rbf", decision_function_shape="ovr", C=1),
    "m4": DecisionTreeClassifier(max_depth=30, criterion="gini"),
}
acc_list_train = {}
acc_list_test = {}
for i in range(40):
    X_train, X_valid, y_train, y_valid = utils.get_X_y(random_state=i)
    for k in model_list.keys():
        model_list[k].fit(X_train, y_train)
        if k not in acc_list_train.keys():
            acc_list_train[k] = []
        if k not in acc_list_test.keys():
            acc_list_test[k] = []
        acc_list_train[k].append(
            accuracy_score(model_list[k].predict(X_train), y_train)
        )
        acc_list_test[k].append(accuracy_score(model_list[k].predict(X_valid), y_valid))

for k in model_list.keys():
    print(
        k, utils.mean_and_std(acc_list_train[k]), utils.mean_and_std(acc_list_test[k])
    )


# X_test = utils.get_X_y_test()
# utils.data_to_csv(model_bnb.predict(X_test), ["id", "cuisine"], path="res.csv")
