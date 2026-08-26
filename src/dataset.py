import torch
from src.tokenizer import CharTokenizer


class CharDataset:
    def __init__(self, text, context_size=16):
        self.tokenizer = CharTokenizer(text)
        self.tokens = self.tokenizer.encode(text)

        self.context_size = context_size

        self.x = []
        self.y = []

        for i in range(len(self.tokens) - context_size):
            context = self.tokens[i:i + context_size]
            target = self.tokens[i + context_size]

            self.x.append(context)
            self.y.append(target)

        self.x = torch.tensor(
            self.x,
            dtype=torch.long
        )

        self.y = torch.tensor(
            self.y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    with open(
        "data/raw/train.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    dataset = CharDataset(text)

    print("Dataset size:", len(dataset))
    print(
        "Vocabulary size:",
        len(dataset.tokenizer.stoi)
    )

    for i in range(10):
        x, y = dataset[i]

        print(
            f"Sample {i}: "
            f"context={x.tolist()} "
            f"target={y.item()}"
        )
