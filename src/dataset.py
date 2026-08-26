import torch
from src.tokenizer import CharTokenizer


class CharDataset:
    def __init__(self, text):
        self.tokenizer = CharTokenizer(text)
        self.tokens = self.tokenizer.encode(text)

        self.x = torch.tensor(self.tokens[:-1], dtype=torch.long)
        self.y = torch.tensor(self.tokens[1:], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    with open("data/raw/train.txt", "r", encoding="utf-8") as f:
        text = f.read()

    dataset = CharDataset(text)

    print("Dataset size:", len(dataset))
    print("Vocabulary size:", len(dataset.tokenizer.stoi))

    for i in range(10):
        x, y = dataset[i]

        print(
            f"Sample {i}: "
            f"input={x.item()} "
            f"target={y.item()}"
        )
