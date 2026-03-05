import torch
import torch.nn as nn


class GRUCTCNet(nn.Module):
    """
    Simple BiGRU encoder + linear classifier for CTC.
    Expects input as (B, T, F).
    Produces logits as (T, B, C) for torch.nn.CTCLoss.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.0,
        fc_features=(384, 384),
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        layers = []
        prev = out_dim
        for h in fc_features:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        y, _ = self.gru(x)               # (B, T, H*)
        y = self.mlp(y)                  # (B, T, H')
        logits = self.classifier(y)      # (B, T, C)
        return logits.transpose(0, 1)    # (T, B, C) for CTC
