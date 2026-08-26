import torch
import torch.nn as nn


class ContextModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=16,
        embedding_dim=32,
        hidden_dim=256
    ):
        super().__init__()

        self.context_size = context_size

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.linear1 = nn.Linear(
            context_size * embedding_dim,
            hidden_dim
        )

        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(
            hidden_dim,
            128
        )

        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(
            128,
            vocab_size
        )

    def forward(self, x):
        embeddings = self.embedding(x)

        embeddings = embeddings.view(
            embeddings.size(0),
            -1
        )

        hidden = self.linear1(embeddings)
        hidden = self.relu1(hidden)

        hidden = self.linear2(hidden)
        hidden = self.relu2(hidden)

        logits = self.linear3(hidden)

        return logits


if __name__ == "__main__":
    vocab_size = 48
    context_size = 16

    model = ContextModel(
        vocab_size=vocab_size,
        context_size=context_size
    )

    x = torch.tensor([
        [23, 31, 36, 3, 26, 30, 24, 37,
         5, 2, 23, 31, 36, 3, 26, 30]
    ])

    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )
