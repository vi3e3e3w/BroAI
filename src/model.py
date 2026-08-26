import torch
import torch.nn as nn


class ContextModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=16,
        embedding_dim=32,
        hidden_dim=128
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

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):
        embeddings = self.embedding(x)

        embeddings = embeddings.view(
            embeddings.size(0),
            -1
        )

        hidden = self.linear1(embeddings)

        hidden = self.relu(hidden)

        logits = self.linear2(hidden)

        return logits


if __name__ == "__main__":
    vocab_size = 46

    model = ContextModel(vocab_size)

    x = torch.tensor([
        [
            21, 29, 34, 1,
            24, 28, 22, 35,
            3, 0, 21, 29,
            34, 1, 24, 28
        ]
    ])

    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("Parameters:", sum(
        p.numel()
        for p in model.parameters()
    ))
