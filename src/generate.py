import torch

from src.tokenizer import CharTokenizer
from src.model import ContextModel


# =========================
# Configuration
# =========================

CONTEXT_SIZE = 16
MAX_NEW_TOKENS = 100

MODEL_PATH = "checkpoints/model.pt"
DATA_PATH = "data/processed/train.txt"


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
        pad_id = tokenizer.stoi["<PAD>"]

        padding = torch.full(
            (CONTEXT_SIZE - len(x),),
            pad_id,
            dtype=torch.long
        )

        x = torch.cat([
            padding,
            x
        ])
    generated_ids = []
    for _ in range(max_new_tokens):

        context = x[-CONTEXT_SIZE:].unsqueeze(0)

        with torch.no_grad():
            logits = model(context)

        last_logits = logits[0]

        temperature = 0.7
        top_k = 5

        repetition_penalty = 1.15

        for token_id in set(x[-CONTEXT_SIZE:].tolist()):
            if last_logits[token_id] > 0:
                last_logits[token_id] /= repetition_penalty
            else:
                last_logits[token_id] *= repetition_penalty

        scaled_logits = last_logits / temperature

        top_values, top_indices = torch.topk(
            scaled_logits,
            k=min(top_k, scaled_logits.size(-1))
        )

        probabilities = torch.softmax(
            top_values,
            dim=0
        )

        sample_index = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        next_id = top_indices[sample_index].item()
        generated_ids.append(next_id)
        x = torch.cat([
            x,
            torch.tensor([next_id], dtype=torch.long)
        ])

        # Stop when the model starts a new speaker.
        generated_text = tokenizer.decode(
            x[original_length:].tolist()
        )

        if "You:" in generated_text:
            break

        if generated_text.count("AI:") > 0:
            break
    # Remove padding and prompt.
    if original_length < CONTEXT_SIZE:
        generated = x[CONTEXT_SIZE:]
    else:
        generated = x[original_length:]

    return tokenizer.decode(
        generated.tolist()
    )

    # Remove accidental speaker labels.
    response = response.replace("AI:", "")

    if "You:" in response:
        response = response.split("You:", 1)[0]

    return response.strip()


# =========================
# Main
# =========================

def main():

    print("=== Cleverbot LLM Edition ===")
    print("Model: ContextModel")
    print(f"Context size: {CONTEXT_SIZE}")
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
            response = generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS
            )

            print("AI:", response)

        except KeyError as e:
            print(
                f"AI: I don't know this character yet: {e}"
            )


if __name__ == "__main__":
    main()
