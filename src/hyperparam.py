from sklearn.model_selection import KFold, train_test_split
from config import DEVICE, DATA_DIR, OUT_DIR, BATCH_SIZE
from torch.utils.data import DataLoader
from model import FFNN, loss_lookup
from train import train_model
from torch.optim import SGD
from utils import load_npz
import numpy as np
import optuna
import torch
import os


def build_model(hparams):
    return FFNN(
        (hparams["first_width"], hparams["first_depth"]),
        (hparams["second_width"], hparams["second_depth"]),
        hparams["dropout"],
    ).to(DEVICE)


def build_optimizer(model, hparams):
    return SGD(
        model.parameters(),
        lr=hparams["learning_rate"],
        momentum=hparams["momentum"],
        weight_decay=hparams["weight_decay"],
    )


def make_predictions(model, emb1s, emb2s):
    predictions = []
    for emb1, emb2 in zip(emb1s, emb2s):
        emb1 = torch.Tensor(emb1).to(DEVICE)
        emb2 = torch.Tensor(emb2).to(DEVICE)
        emb1_sentiment = model(emb1).item()
        emb2_sentiment = model(emb2).item()
        predictions.append(0 if emb1_sentiment > emb2_sentiment else 1)

    return predictions


def evaluate_model(model, emb1s, emb2s, labels):
    predictions = make_predictions(model, emb1s, emb2s)
    return np.equal(labels, predictions).mean()


def cross_validate(data, hparams, trial=None):
    emb1, emb2, labels = data

    criterion, dataset = loss_lookup[hparams["loss"]]

    kfold = KFold(n_splits=5, shuffle=True)
    test_scores = []

    for step, (train_indices, test_indices) in enumerate(kfold.split(emb1)):
        train_dataset = dataset(
            emb1[train_indices], emb2[train_indices], labels[train_indices]
        )
        train_dataset, val_dataset = train_test_split(
            train_dataset, train_size=hparams["train_val_split"], shuffle=True
        )
        model = build_model(hparams)
        optimizer = build_optimizer(model, hparams)

        model = train_model(
            model,
            optimizer,
            criterion,
            DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True),
            hparams["num_epochs"],
            graph=True,
        )

        test_scores.append(
            evaluate_model(
                model, emb1[test_indices], emb2[test_indices], labels[test_indices]
            )
        )

        kf_mean = np.mean(test_scores)

        if trial:
            trial.report(kf_mean, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return kf_mean


def objective(data, trial):
    hparams = {
        "loss": trial.suggest_categorical("loss", loss_lookup.keys()),
        "num_epochs": trial.suggest_int("num_epochs", 20, 200),
        "train_val_split": trial.suggest_float("train_val_split", 0.05, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-1, log=True),
        "momentum": trial.suggest_float("momentum", 1e-2, 1),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-1, log=True),
        "first_width": trial.suggest_float("first_width", 0.5, 3),
        "first_depth": trial.suggest_int("first_depth", 0, 10),
        "second_width": trial.suggest_float("second_width", 0.125, 2),
        "second_depth": trial.suggest_int("second_depth", 0, 10),
        "dropout": trial.suggest_float("dropout", 0, 0.75),
    }

    return cross_validate(data, hparams, trial)


if __name__ == "__main__":
    train_data = load_npz(os.path.join(DATA_DIR, "train.npz"))
    data = (
        train_data["emb1"],
        train_data["emb2"],
        train_data["preference"],
    )
    name = "cs4780"

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{OUT_DIR}cs4780.db",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(
        lambda trial: objective(data, trial),
        n_trials=500,
    )
