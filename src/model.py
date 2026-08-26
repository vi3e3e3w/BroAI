import torch
import torch.nn as nn


class BigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.linear = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        logits = self.linear(embeddings)

        return logits


if __name__ == "__main__":
    vocab_size = 27

    model = BigramModel(vocab_size)

    x = torch.tensor([10, 18, 21])

    logits = model(x)

    print("Input:", x)
    print("Logits shape:", logits.shape)
    print("Logits:")
    print(logits)
