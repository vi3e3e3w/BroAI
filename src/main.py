import torch

from src.tokenizer import CharTokenizer
from src.model import ContextModel
from src.generate import generate


# =========================
# Configuration
# =========================

CONTEXT_SIZE = 16
MAX_NEW_TOKENS = 100

MODEL_PATH = "checkpoints/model.pt"
DATA_PATH = "data/processed/train.txt"
CONVERSATION_PATH = "data/conversations/chat.txt"

# =========================
# Load training data
# =========================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()


# =========================
# Tokenizer
# =========================

tokenizer = CharTokenizer(text)

vocab_size = len(tokenizer.stoi)


# =========================
# Load model
# =========================

model = ContextModel(
    vocab_size=vocab_size,
    context_size=CONTEXT_SIZE
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# =========================
# Text generation
# =========================

def generate(
    prompt,
    max_new_tokens=MAX_NEW_TOKENS
):
    ids = tokenizer.encode(prompt)

    x = torch.tensor(
        ids,
        dtype=torch.long
    )

    original_length = len(x)

    # ContextModel requires exactly
    # CONTEXT_SIZE tokens.
    if len(x) < CONTEXT_SIZE:
        padding = torch.zeros(
            CONTEXT_SIZE - len(x),
            dtype=torch.long
        )

        x = torch.cat([
            padding,
            x
        ])

    for _ in range(max_new_tokens):

        # Use only the latest context.
        context = x[-CONTEXT_SIZE:].unsqueeze(0)

        with torch.no_grad():
            logits = model(context)

        # Model output:
        # [batch, vocab_size]
        last_logits = logits[0]

        probabilities = torch.softmax(
            last_logits,
            dim=0
        )

        # Sample instead of always
        # choosing the most likely token.
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

    # Remove padding + original prompt.
    if original_length < CONTEXT_SIZE:
        generated = x[CONTEXT_SIZE:]
    else:
        generated = x[original_length:]

    return tokenizer.decode(
        generated.tolist()
    )


# =========================
# Main
# =========================

def main():
    print("=== Cleverbot LLM Edition ===")
    print("Model: ContextModel")
    print("Context size: 16")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("You: ")

        except (KeyboardInterrupt, EOFError):
            print("\nAI: Bye!")
            break

        if prompt.lower() == "exit":
            print("AI: Bye!")
            break

        if not prompt:
            continue

        try:
            tagged_prompt = f"<USER>{prompt}\n<AI>"

            response = generate(
                 tagged_prompt,
                 max_new_tokens=MAX_NEW_TOKENS
            )

            response = response.strip()

            if response.startswith("AI:"):
                response = response[3:].lstrip()

            print("AI:", response)

            # Save conversation
            with open(
                CONVERSATION_PATH,
                "a",
                encoding="utf-8"
            ) as f:
                f.write(f"{prompt}\n")
                f.write(f"{response}\n")
                f.write("\n")

        except KeyError as e:
            print(
                f"AI: I don't know this character yet: {e}"
            )
if __name__ == "__main__":
    main()
