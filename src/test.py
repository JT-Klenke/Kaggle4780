from hyperparam import build_model, build_optimizer, evaluate_model, make_predictions
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from config import DATA_DIR, OUT_DIR
from model import loss_lookup
from train import train_model
from utils import load_npz
import torch
import csv
import os


def make_csv(model, test_data, csv_name="answers"):
    uids = test_data["uids"]
    emb1 = test_data["emb1"]
    emb2 = test_data["emb2"]

    predictions = list(zip(uids, make_predictions(model, emb1, emb2)))
    csv_file_name = os.path.join(OUT_DIR, f"{csv_name}.csv")

    with open(csv_file_name, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["uid", "preference"])
        csv_writer.writerows(predictions)


if __name__ == "__main__":
    hparams = {}

    train_data = load_npz(os.path.join(DATA_DIR, "train.npz"))
    test_data = load_npz(os.path.join(DATA_DIR, "test.npz"))

    criterion, dataset = loss_lookup[hparams["loss"]]

    (
        train_emb1,
        test_emb1,
        train_emb2,
        test_emb2,
        train_labels,
        test_labels,
    ) = train_test_split(
        train_data["emb1"],
        train_data["emb2"],
        train_data["preference"],
        test_size=0.2,
        shuffle=True,
    )

    train_dataset = dataset(train_emb1, train_emb2, train_labels)
    train_dataset, val_dataset = train_test_split(
        train_dataset, train_size=hparams["train_val_split"], shuffle=True
    )

    model = build_model(hparams)
    optimizer = build_optimizer(model, hparams)

    model = train_model(
        model,
        optimizer,
        criterion,
        DataLoader(train_dataset, batch_size=512, shuffle=True),
        DataLoader(val_dataset, batch_size=512, shuffle=True),
        hparams["num_epochs"],
        graph=False,
    )

    print(evaluate_model(model, test_emb1, test_emb2, test_labels))

    torch.save(model, os.path.join(OUT_DIR, "tested_model.pth"))

    make_csv(model, test_data)
