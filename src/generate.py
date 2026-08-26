import torch

from src.tokenizer import CharTokenizer
from src.model import BigramModel


# Load training data
with open("data/raw/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

# Create model with the same architecture used during training
model = BigramModel(
    vocab_size=len(tokenizer.stoi)
)

# Load trained weights
model.load_state_dict(
    torch.load(
        "checkpoints/model.pt",
        map_location="cpu"
    )
)

model.eval()


def generate(prompt, max_new_tokens=100):
    ids = tokenizer.encode(prompt)

    x = torch.tensor(ids, dtype=torch.long)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x)

        # Prediction for the last character
        last_logits = logits[-1]

        # Convert logits to probabilities
        probabilities = torch.softmax(last_logits, dim=0)

        # Pick the most likely character
        next_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        # Append it
        x = torch.cat([
            x,
            torch.tensor([next_id], dtype=torch.long)
        ])

    return tokenizer.decode(x.tolist())


if __name__ == "__main__":
    print("=== Cleverbot LLM Edition ===")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() == "exit":
            print("AI: Bye!")
            break

        try:
            output = generate(
                prompt,
                max_new_tokens=100
            )

            print("AI:", output)

        except KeyError as e:
            print("AI: Unknown character:", e)
