from torch import nn


class FFNN(nn.Module):
    def __init__(self, first_section, second_section, final_activation, dropout):
        super().__init__()
        first_width, first_depth = first_section
        second_width, second_depth = second_section
        first_width = int(first_width * 384)
        second_width = int(second_width * 384)

        self.first_layer = nn.Sequential(
            nn.Linear(384, first_width),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )

        self.first_section = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(first_width, first_width),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout),
                )
                for _ in range(first_depth)
            ]
        )

        self.transition_layer = nn.Sequential(
            nn.Linear(first_width, second_width),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )

        self.second_section = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(first_width, first_width),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout),
                )
                for _ in range(second_depth)
            ]
        )

        self.last_layer = nn.Sequential(nn.Linear(second_width, 1), final_activation)

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
