from dataset import Embedding01, EmbeddingPM
import torch.nn.functional as F
from torch import nn
import torch


class FFNN(nn.Module):
    def __init__(self, first_section, second_section, dropout):
        super().__init__()
        first_width, first_depth, first_norm = first_section
        second_width, second_depth, second_norm = second_section
        first_width = int(first_width * 384)
        second_width = int(second_width * 384)

        self.first_layer = nn.Sequential(
            nn.Linear(384, first_width),
            nn.Dropout(p=dropout),
        )

        self.first_section = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(first_width) if first_norm else nn.Identity(),
                    nn.LeakyReLU(),
                    nn.Linear(first_width, first_width),
                    nn.BatchNorm1d(first_width) if first_norm else nn.Identity(),
                    nn.LeakyReLU(),
                    nn.Linear(first_width, first_width),
                    nn.Dropout(p=dropout),
                )
                for _ in range(first_depth)
            ]
        )

        self.transition_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(first_width, second_width),
            nn.Dropout(p=dropout),
        )

        self.second_section = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(second_width) if second_norm else nn.Identity(),
                    nn.LeakyReLU(),
                    nn.Linear(second_width, second_width),
                    nn.BatchNorm1d(second_width) if second_norm else nn.Identity(),
                    nn.LeakyReLU(),
                    nn.Linear(second_width, second_width),
                    nn.Dropout(p=dropout),
                )
                for _ in range(second_depth)
            ]
        )

        self.last_layer = nn.Sequential(nn.LeakyReLU(), nn.Linear(second_width, 1))

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.first_section:
            residual = x
            x = layer(x)
            x += residual

        x = self.transition_layer(x)

        for layer in self.second_section:
            residual = x
            x = layer(x)
            x += residual

        return self.last_layer(x)


bce_loss = nn.BCEWithLogitsLoss()
mse_loss = lambda pred, true: F.mse_loss(F.sigmoid(pred), true)
hinge_loss = lambda pred, true: F.relu(0.25 - pred * true).mean()
leaky_hinge_loss = lambda pred, true: F.leaky_relu(0.25 - pred * true).mean()
exp_loss = lambda pred, true: torch.exp(-true * pred).mean()


loss_lookup = {
    "bce": (bce_loss, Embedding01),
    "mse": (mse_loss, Embedding01),
    "hinge": (hinge_loss, EmbeddingPM),
    "leaky_hinge": (leaky_hinge_loss, EmbeddingPM),
    "exp": (exp_loss, EmbeddingPM),
}
