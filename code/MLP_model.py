import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
from typing import Union
import numpy as np
from sklearn.metrics import accuracy_score

map_dict = {"indian": 0, "chinese": 1, "korean": 2, "thai": 3, "japanese": 4}


@torch.no_grad()
def model_eval(
    model: torch.nn.Module,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss] = None,
):
    logit = model(torch.from_numpy(X_eval.astype(np.float64)))
    pred = torch.softmax(logit, -1).argmax(-1)
    acc = accuracy_score(y_true=y_eval, y_pred=pred)
    loss = None
    if criterion is not None:
        loss = criterion(logit, torch.from_numpy(y_eval))
    return acc, loss


def model_train(
    rounds: int,
    eta: float,
    model: torch.nn.Module,
    criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
    batch_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_while_training: bool = False,
):
    acc_list = []
    loss_list = []
    idx_all = list(range(len(X_train)))
    batch_pointer = 0
    for _ in range(rounds):
        new_lst = idx_all[batch_pointer : batch_pointer + batch_size]
        if batch_pointer + batch_size > len(X_train):
            new_lst = idx_all[batch_pointer : len(X_train)]
        x = torch.from_numpy(X_train[new_lst].astype(np.float64))
        y = torch.from_numpy(y_train[new_lst])
        batch_pointer += batch_size
        if batch_pointer >= len(X_train):
            batch_pointer = 0
        logit = model(x)
        loss = criterion(logit, y)
        grads = torch.autograd.grad(loss, model.parameters())
        for param, grad in zip(model.parameters(), grads):
            param.data.sub_(eta * grad)

        if eval_while_training:
            acc_, loss_ = model_eval(model, X_eval, y_eval, criterion)
            acc_list.append(acc_)
            loss_list.append(loss_)
    return acc_list, loss_list


class elu(nn.Module):
    def __init__(self) -> None:
        super(elu, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x, 0.2 * (torch.exp(x) - 1))


class linear(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(linear, self).__init__()
        self.w = nn.Parameter(
            torch.randn(out_c, in_c) * torch.sqrt(torch.tensor(2 / in_c))
        )
        self.b = nn.Parameter(torch.randn(out_c))
        self.double()

    def forward(self, x):
        return F.linear(x, self.w, self.b)


class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.fc1 = linear(383, 40)
        self.fc2 = linear(40, 10)
        self.fc3 = linear(10, 5)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x


if __name__ == "__main__":
    acc_train_list = []
    acc_test_list = []

    acc_training_history_all = []
    loss_training_history_all = []
    total_round = 40
    for i in range(total_round):
        X_train, X_valid, y_train, y_valid = utils.get_X_y(random_state=i)
        model = MLP()
        acc_training_history, loss_training_history = model_train(
            15000,
            0.007,
            model,
            torch.nn.CrossEntropyLoss(),
            40,
            X_train,
            np.array(
                utils.str_map_2_int(
                    y_train,
                    map_dict,
                )
            ),
            X_valid,
            np.array(
                utils.str_map_2_int(
                    y_valid,
                    map_dict,
                )
            ),
            True,
        )
        acc_training_history_all.append(acc_training_history)
        loss_training_history_all.append(loss_training_history)
        acc_train_list.append(
            model_eval(
                model,
                X_train,
                np.array(utils.str_map_2_int(y_train, map_dict)),
            )[0]
        )

        acc_test_list.append(
            model_eval(
                model,
                X_valid,
                np.array(utils.str_map_2_int(y_valid, map_dict)),
            )[0]
        )
    acc_training_history_all = np.array(acc_training_history_all).T
    loss_training_history_all = np.array(loss_training_history_all).T
    utils.data_to_csv(
        acc_training_history_all,
        utils.link_list([["round"], [str(i + 1) for i in range(total_round)]]),
        "acc_history.csv",
    )
    utils.data_to_csv(
        loss_training_history_all,
        utils.link_list([["round"], [str(i + 1) for i in range(total_round)]]),
        "loss_history.csv",
    )

    # X_test = utils.get_X_y_test()
    # logit = model(torch.from_numpy(X_test.astype(np.float64)))
    # pred = torch.softmax(logit, -1).argmax(-1)
    # utils.data_to_csv(
    #     utils.int_map_2_str(np.array(pred), map_dict), ["id", "cuisine"], "res.csv"
    # )
