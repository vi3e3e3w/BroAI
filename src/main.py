import torch

from src.tokenizer import CharTokenizer
from src.model import BigramModel


# =========================
# Load training data
# =========================

with open("data/raw/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)


# =========================
# Load trained model
# =========================

model = BigramModel(
    vocab_size=len(tokenizer.stoi)
)

model.load_state_dict(
    torch.load(
        "checkpoints/model.pt",
        map_location="cpu"
    )
)

model.eval()


# =========================
# Text generation
# =========================

def generate(prompt, max_new_tokens=100):
    ids = tokenizer.encode(prompt)

    x = torch.tensor(
        ids,
        dtype=torch.long
    )

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x)

        # Get prediction for the last character
        last_logits = logits[-1]

        # Convert logits to probabilities
        probabilities = torch.softmax(
            last_logits,
            dim=0
        )

        # Sample the next character
        next_id = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        x = torch.cat([
            x,
            torch.tensor(
                [next_id],
                dtype=torch.long
            )
        ])

    return tokenizer.decode(x.tolist())


# =========================
# Main
# =========================

def main():
    print("=== Cleverbot LLM Edition ===")
    print("Model: BigramModel")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("You: ")

        if prompt.lower() == "exit":
            print("AI: Bye!")
            break

        if not prompt:
            continue

        try:
            output = generate(
                prompt,
                max_new_tokens=100
            )

            # Remove the original prompt from the output
            response = output[len(prompt):]

            print("AI:", response)

        except KeyError as e:
            print(
                f"AI: I don't know this character yet: {e}"
            )


if __name__ == "__main__":
    main()

