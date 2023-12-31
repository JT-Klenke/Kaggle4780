from matplotlib import pyplot as plt
from config import DEVICE, OUT_DIR
from torch.optim import SGD
import numpy as np
import torch
import os


def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    losses = []
    for embedding, label in dataloader:
        optimizer.zero_grad()
        output = model(embedding.to(DEVICE))
        loss = criterion(output, label.to(DEVICE))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def validate_epoch(model, criterion, dataloader):
    model.eval()
    losses = []
    for embedding, label in dataloader:
        with torch.no_grad():
            output = model(embedding.to(DEVICE))
            loss = criterion(output, label.to(DEVICE))
        losses.append(loss.item())

    return losses


def plot(losses, color, save_path):
    _, ax = plt.subplots()

    epochs = np.arange(0, len(losses))
    ax.plot(epochs, losses, linestyle="-", color=color)

    plt.savefig(save_path, format="png")
    plt.close()


def train_model(
    model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    num_epochs=100,
    graph=True,
):
    train_losses = []
    val_losses = []
    best_val = float("inf")

    temp_optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(num_epochs):
        train_losses.append(
            np.mean(
                train_epoch(
                    model,
                    optimizer if epoch > 5 else temp_optimizer,
                    criterion,
                    train_dataloader,
                )
            )
        )
        val_loss = np.mean(validate_epoch(model, criterion, val_dataloader))
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model, os.path.join(OUT_DIR, "best_val.pth"))
        if graph:
            plot(train_losses, "blue", os.path.join(OUT_DIR, "train_losses.png"))
            plot(val_losses, "red", os.path.join(OUT_DIR, "val_losses.png"))

    return torch.load(os.path.join(OUT_DIR, "best_val.pth")).to(DEVICE)
