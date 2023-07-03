from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import utils
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import copy


def stacking_predict(model_list, agg_model, X):
    ys_array = []
    for model in model_list:
        ys = model.predict(X)
        ys = utils.str_map_2_int(
            ys, {"indian": 0, "chinese": 1, "korean": 2, "thai": 3, "japanese": 4}
        )
        ys_array.append(ys)
    ys_array = np.array(ys_array).T
    return agg_model.predict(ys_array)


def bagging_predict(model_list, X):
    def select_max_count(ll):
        count_dict = {}
        for v in ll:
            if v not in count_dict.keys():
                count_dict[v] = 1
            else:
                count_dict[v] += 1
        max_v = -1
        max_k = None
        for k in count_dict.keys():
            if max_v < count_dict[k]:
                max_k = k
                max_v = count_dict[k]
        return max_k

    ys_array = []
    for model in model_list:
        ys = model.predict(X)
        ys_array.append(ys)

    ys_array = np.array(ys_array).T
    res = []
    for arr in ys_array:
        res.append(select_max_count(list(arr)))
    return res


def split_data(data_num, shard_fractions: list):
    if sum([s * 10 for s in shard_fractions]) != 10:
        print(sum(shard_fractions))
        print("The sum of shard fractions should be 1")
        return []
    data_id_list = [i for i in range(data_num)]
    shards = []
    for i in range(len(shard_fractions)):
        sample_shard_num = int(data_num * shard_fractions[i])
        if i == len(shard_fractions) - 1:
            sample_shard_num = len(data_id_list)
        shard = random.sample(data_id_list, sample_shard_num)
        shards.append(shard)
        for s in shard:
            data_id_list.remove(s)
    return shards


if __name__ == "__main__":
    acc_list_train = []
    acc_list_test = []
    acc_list_train_1 = []
    acc_list_test_1 = []
    basic_model_acc = {}
    basic_model_acc_1 = {}
    for train_idx in range(40):
        X_train, X_valid, y_train, y_valid = utils.get_X_y(random_state=train_idx)
        model_list = {
            "m1": LogisticRegression(penalty="l1", C=0.5, solver="liblinear"),
            # "m2": LogisticRegression(penalty="l2", C=0.5, solver="liblinear"),
            "m3": BernoulliNB(alpha=1.5),
            "m5": SVC(kernel="rbf", decision_function_shape="ovr", C=1),
        }
        model_list_1 = copy.deepcopy(model_list)
        data_shards = split_data(len(X_train), [0.9, 0.1])
        for k in model_list.keys():
            model_list[k].fit(X_train[data_shards[0]], y_train[data_shards[0]])
            model_list_1[k].fit(X_train, y_train)

        ys_array = []
        for k in model_list.keys():
            ys = model_list[k].predict(X_train[data_shards[-1]])
            ys = utils.str_map_2_int(
                ys, {"indian": 0, "chinese": 1, "korean": 2, "thai": 3, "japanese": 4}
            )
            ys_array.append(ys)
        ys_array = np.array(ys_array).T

        meta_model = DecisionTreeClassifier()
        meta_model.fit(ys_array, y_train[data_shards[-1]])

        # print(ys_array)
        acc_list_train.append(
            accuracy_score(
                stacking_predict(
                    [model_list[k] for k in model_list.keys()], meta_model, X_train
                ),
                y_train,
            )
        )
        acc_list_test.append(
            accuracy_score(
                stacking_predict(
                    [model_list[k] for k in model_list.keys()], meta_model, X_valid
                ),
                y_valid,
            )
        )
        acc_list_train_1.append(
            accuracy_score(
                bagging_predict([model_list_1[k] for k in model_list.keys()], X_train),
                y_train,
            )
        )
        acc_list_test_1.append(
            accuracy_score(
                bagging_predict([model_list_1[k] for k in model_list.keys()], X_valid),
                y_valid,
            )
        )
        for k in model_list.keys():
            if k not in basic_model_acc.keys():
                basic_model_acc[k] = []
                basic_model_acc[k].append(
                    accuracy_score(model_list[k].predict(X_valid), y_valid)
                )
                basic_model_acc_1[k] = []
                basic_model_acc_1[k].append(
                    accuracy_score(model_list_1[k].predict(X_valid), y_valid)
                )
            else:
                basic_model_acc[k].append(
                    accuracy_score(model_list[k].predict(X_valid), y_valid)
                )
                basic_model_acc_1[k].append(
                    accuracy_score(model_list_1[k].predict(X_valid), y_valid)
                )
        # for model in model_list:
    print(utils.mean_and_std(acc_list_train), utils.mean_and_std(acc_list_test))
    print(utils.mean_and_std(acc_list_train_1), utils.mean_and_std(acc_list_test_1))
    for k in basic_model_acc.keys():
        print(k, ":", utils.mean_and_std(basic_model_acc[k]))
    for k in basic_model_acc_1.keys():
        print(k, ":", utils.mean_and_std(basic_model_acc_1[k]))
