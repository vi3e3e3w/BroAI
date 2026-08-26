import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import CharDataset
from src.model import ContextModel


# =========================
# 1. Load training data
# =========================

with open("data/raw/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

context_size = 16

dataset = CharDataset(
    text,
    context_size=context_size
)

vocab_size = len(dataset.tokenizer.stoi)

print("Dataset size:", len(dataset))
print("Vocabulary size:", vocab_size)


# =========================
# 2. Create model
# =========================

model = ContextModel(
    vocab_size=vocab_size,
    context_size=context_size
)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)
# =========================
# 3. Loss + optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001
)


# =========================
# 4. Training loop
# =========================

epochs = 1000

for epoch in range(epochs):

    x = dataset.x
    y = dataset.y

    logits = model(x)

    loss = criterion(
        logits,
        y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss.item():.4f}"
        )


# =========================
# 5. Save model
# =========================

torch.save(
    model.state_dict(),
    "checkpoints/model.pt"
)

print("Training complete!")
print("Saved: checkpoints/model.pt")
