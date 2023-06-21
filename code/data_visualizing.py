import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import utils
from collections import Counter
from sklearn import tree
from sklearn.tree._tree import Tree
import graphviz

# more params: https://drmattcrooks.medium.com/how-to-set-up-rcparams-in-matplotlib-355a0b9494ec
sns.set(
    rc={
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "grid.color": "black",
        "axes.grid.axis": "x",
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.edgecolor": "grey",
        "axes.linewidth": 0.2,
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
        "patch.edgecolor": "w",
    }
)
# sns.set_style("white")

# sns.set()
SimSun = FontProperties(fname="/Users/liuyuxiao/.fonts/SimSun.ttf", size=12)
ori_data = pd.read_csv("../data/recipes_train.csv")
feature_names = list(ori_data.columns)[2:]
data_labels = list(ori_data["cuisine"])
label_names = utils.deduplicate(data_labels)
path = "../report/contents/figs_and_tables/"
label_count = dict(Counter(data_labels))
label_count["all"] = len(data_labels)


def trait_label_relationship():
    label_names_sort: list = label_names.copy()
    label_names_sort.sort()
    path_df_columns = utils.link_list([["path_id"], ["leaf_node"], feature_names])
    path_df_all = None
    for tree_num in range(100):
        dtc = tree.DecisionTreeClassifier(criterion="gini")
        dtc = dtc.fit(np.array(ori_data[feature_names]), list(ori_data["cuisine"]))
        tree.export_graphviz(
            dtc,  # 训练好的模型
            out_file="dot.txt",
            feature_names=feature_names,
            class_names=label_names_sort,
            filled=True,  # 进行颜色填充
            rounded=True,  # 树节点的形状控制
        )
        dot_data = None
        with open("dot.txt", "r", encoding="utf-8") as f:  # 打开文本
            dot_data = f.read()  # 读取文本
        tree_path_list = utils.extract_tree_from_dot(ori_data=dot_data)

        path_df = pd.DataFrame(columns=path_df_columns)

        for idx in range(len(tree_path_list)):
            tmp_path = tree_path_list[idx].split(",")
            data_in_path_df = [idx, tmp_path[-2]]
            for feature in feature_names:
                if feature in tmp_path:
                    data_in_path_df.append(1)
                else:
                    data_in_path_df.append(0)
            path_df.loc[len(path_df.index)] = data_in_path_df
        path_df = path_df.groupby(["leaf_node"]).agg(
            {feature: "sum" for feature in feature_names}
        )
        path_df = path_df.reset_index()
        path_df["tree_num"] = tree_num
        path_df["col_sum"] = path_df[feature_names].apply(
            lambda x: sum([x[f] for f in feature_names]), axis=1
        )
        # print(path_df)
        for f in feature_names:
            path_df[f] = path_df[f] / path_df["col_sum"]
        if path_df_all is None:
            path_df_all = path_df
        else:
            path_df_all = pd.concat([path_df_all, path_df], axis=0)
    print(path_df_all)
    # print(path_df.loc["chinese"].argmax(), path_df.loc["chinese"].max())
    # print(path_df.loc["chinese"])
    # print(path_df.loc["chinese"][-1])


def label_distribution():
    fig: sns.FacetGrid = sns.displot(data_labels, shrink=0.6)
    fig._figure.set_figwidth(10)
    fig._figure.set_figheight(4)
    fig._figure.axes[0].spines["top"].set_visible(True)
    fig._figure.axes[0].spines["right"].set_visible(True)
    fig.set_ylabels("数量", fontproperties=SimSun)
    fig.savefig(
        path + "data_visualize/label_distribution.pdf",
        dpi=500,
        pad_inches=0,
        bbox_inches="tight",
    )


def feature_sparsity():
    ori_data["default"] = "all"
    ori_data["nonzero_feature"] = ori_data[feature_names].apply(
        lambda x: sum([1 if x[name] != 0 else 0 for name in feature_names]), axis=1
    )
    sparse_rate_4_samples: list = [
        v / len(feature_names) for v in ori_data["nonzero_feature"]
    ]
    sparse_rate_dict = dict(Counter(sparse_rate_4_samples))
    total = sum(list(sparse_rate_dict.values()))
    larger_05 = sum(
        [sparse_rate_dict[key] if key > 0.05 else 0 for key in sparse_rate_dict]
    )
    print(total, larger_05, (total - larger_05) / total, larger_05 / total)
    fig1, ax1 = plt.subplots()
    fig1.set_figheight(2)
    fig1.set_figwidth(10)
    # ax1.scatter(
    #     range(len(sparse_rate_4_samples)),
    #     sparse_rate_4_samples,
    #     s=[0.1] * len(sparse_rate_4_samples),
    # )
    sparse_rate_4_samples.sort()
    ax1.bar(
        x=range(len(sparse_rate_4_samples)),
        height=sparse_rate_4_samples,
        linewidth=0,
        width=[0.5] * len(sparse_rate_4_samples),
    )
    fig1.savefig(
        path + "data_visualize/sparse_rate_4_samples.pdf",
        dpi=500,
        pad_inches=0,
        bbox_inches="tight",
    )
    sparse_data_4_labels = ori_data.groupby(["cuisine"]).apply(
        lambda x: pd.Series(
            {feature: x[x[feature] != 0][feature].count() for feature in feature_names}
        )
    )
    sparse_data_4_all = ori_data.groupby(["default"]).apply(
        lambda x: pd.Series(
            {feature: x[x[feature] != 0][feature].count() for feature in feature_names}
        )
    )
    sparse_data = pd.concat([sparse_data_4_all, sparse_data_4_labels], axis=0)
    for label in label_names:
        fig, ax = plt.subplots()
        fig.set_figheight(1)
        fig.set_figwidth(15)

        ax.bar(
            x=range(len(feature_names)),
            height=[each / label_count[label] for each in list(sparse_data.loc[label])],
            width=[0.5] * len(feature_names),
            linewidth=0,
        )
        ax.set_xticks([])
        sparse_data.to_csv("sparse_data.csv")
        fig.savefig(
            path + "data_visualize/frequency_{}.pdf".format(label),
            dpi=500,
            pad_inches=0,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    # label_distribution()
    # feature_sparsity()
    trait_label_relationship()
    # print(sns.axes_style())
    print(label_count)
    print(label_names)
