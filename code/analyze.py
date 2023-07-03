import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

SimSun = FontProperties(fname="/Users/liuyuxiao/.fonts/SimSun.ttf", size=12)
path = "../report/contents/figs_and_tables/"
kpi = {"acc": "准确率", "loss": "损失值"}

if __name__ == "__main__":
    acc_history = pd.read_csv("acc_history.csv")
    acc_cols = list(acc_history.columns)
    acc_history["mean"] = acc_history.apply(
        lambda x: np.array([x[col] for col in acc_cols[1:]]).mean(), axis=1
    )
    loss_history = pd.read_csv("loss_history.csv")
    loss_cols = list(loss_history.columns)
    loss_history["mean"] = loss_history.apply(
        lambda x: np.array([x[col] for col in loss_cols[1:]]).mean(), axis=1
    )
    print(loss_history["mean"])
    for k in kpi.keys():
        fig1, ax1 = plt.subplots()
        fig1.set_figheight(10)
        fig1.set_figwidth(10)
        if k == "loss":
            ax1.plot(list(loss_history["mean"]))
        if k == "acc":
            ax1.plot(list(acc_history["mean"]))
        ax1.set_ylabel(kpi[k], fontproperties=SimSun)
        ax1.set_xlabel("训练轮数", fontproperties=SimSun)
        fig1.savefig(
            dpi=500,
            fname=path + "model_analyze/mlp_{}.pdf".format(k),
            pad_inches=0,
            bbox_inches="tight",
        )
