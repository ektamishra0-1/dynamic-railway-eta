import torch
import torch.nn as nn


class GRUAttention(nn.Module):

    def __init__(
        self,
        numeric_dim,
        categorical_sizes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.25,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(size + 1, 16)
            for size in categorical_sizes
        ])

        embedding_dim = 16 * len(categorical_sizes)
        input_dim = numeric_dim + embedding_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, numeric, categorical):

        embedded = []

        for i, embedding in enumerate(self.embeddings):
            embedded.append(
                embedding(categorical[:, :, i])
            )

        x = torch.cat(
            [numeric] + embedded,
            dim=-1
        )

        gru_out, _ = self.gru(x)

        scores = self.attention(gru_out)

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            weights * gru_out,
            dim=1
        )

        context = self.dropout(context)

        output = self.head(context)

        return output.squeeze(-1)
