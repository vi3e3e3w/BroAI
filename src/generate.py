import torch

from src.tokenizer import CharTokenizer
from src.model import ContextModel


# =========================
# Configuration
# =========================

CONTEXT_SIZE = 16
MAX_NEW_TOKENS = 100
ROLE_TAGS = ("<USER>", "<AI>")
MALFORMED_ROLE_TAGS = ("<AR>",)
ROLE_TAG_PREFIXES = tuple(
    sorted(
        {
            tag[:length]
            for tag in ROLE_TAGS
            for length in range(2, len(tag))
        },
        key=len,
        reverse=True
    )
)

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
# Load model
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_state_dict = checkpoint["model_state_dict"]
    if "stoi" in checkpoint:
        tokenizer = CharTokenizer.from_mappings(
            checkpoint["stoi"],
            checkpoint.get("itos")
        )
    else:
        tokenizer = CharTokenizer(text)
    context_size = checkpoint.get("context_size", CONTEXT_SIZE)
else:
    model_state_dict = checkpoint
    tokenizer = CharTokenizer(text)
    context_size = CONTEXT_SIZE

model = ContextModel(
    vocab_size=len(tokenizer.stoi),
    context_size=context_size
)

model.load_state_dict(
    model_state_dict
)

model.eval()


def _find_role_marker(text, allow_incomplete_tail=False):
    for index, char in enumerate(text):
        if char == "<":
            for tag in ROLE_TAGS:
                if text.startswith(tag, index):
                    return index, index + len(tag), True

            for tag in MALFORMED_ROLE_TAGS:
                if text.startswith(tag, index):
                    before_is_boundary = index == 0 or not (
                        text[index - 1].isalnum() or text[index - 1] == "_"
                    )
                    end = index + len(tag)
                    after_is_boundary = end == len(text) or not (
                        text[end].isalnum() or text[end] == "_"
                    )
                    if before_is_boundary and after_is_boundary:
                        return index, end, False

            for prefix in ROLE_TAG_PREFIXES:
                if not text.startswith(prefix, index):
                    continue

                end = index + len(prefix)
                if end == len(text) and allow_incomplete_tail:
                    return index, end, False
                if end < len(text) and (text[end].isspace() or text[end] == ">"):
                    if text[end] == ">":
                        end += 1
                    return index, end, False

            if text.startswith("<AR", index):
                before_is_boundary = index == 0 or not (
                    text[index - 1].isalnum() or text[index - 1] == "_"
                )
                end = index + len("<AR")
                if end == len(text) and allow_incomplete_tail and before_is_boundary:
                    return index, end, False
                if (
                    before_is_boundary
                    and end < len(text)
                    and (text[end].isspace() or text[end] == ">")
                ):
                    if text[end] == ">":
                        end += 1
                    return index, end, False

        for tag in ROLE_TAGS:
            fragment = tag[1:]
            if not text.startswith(fragment, index):
                continue

            before_is_boundary = index == 0 or not (
                text[index - 1].isalnum() or text[index - 1] == "_"
            )
            end = index + len(fragment)
            after_is_boundary = end == len(text) or not (
                text[end].isalnum() or text[end] == "_"
            )
            if before_is_boundary and after_is_boundary:
                return index, end, False

        if text.startswith("AR>", index):
            before_line = text[text.rfind("\n", 0, index) + 1:index].strip()
            before_text = text[:index].rstrip()
            at_sentence_start = (
                not before_line
                or not before_text
                or before_text[-1] in ".!?"
            )
            end = index + len("AR>")
            after_is_boundary = end == len(text) or text[end].isspace()
            if at_sentence_start and after_is_boundary:
                return index, end, False

    return None


def clean_generated_text(text):
    output = []
    position = 0

    while position < len(text):
        marker = _find_role_marker(
            text[position:],
            allow_incomplete_tail=True
        )
        if marker is None:
            output.append(text[position:])
            break

        start, end, is_complete = marker
        output.append(text[position:position + start])
        if is_complete:
            break

        position += end
        if (
            output
            and output[-1]
            and output[-1][-1].isspace()
            and position < len(text)
            and text[position].isspace()
        ):
            position += 1

    return "".join(output).strip()


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

    # ContextModel requires exactly context_size tokens.
    if len(x) < context_size:
        pad_id = tokenizer.stoi["<PAD>"]

        padding = torch.full(
            (context_size - len(x),),
            pad_id,
            dtype=torch.long
        )

        x = torch.cat([
            padding,
            x
        ])

    generated_ids = []

    temperature = 0.7
    top_k = 5
    repetition_penalty = 1.15

    for _ in range(max_new_tokens):

        context = x[-context_size:].unsqueeze(0)

        with torch.no_grad():
            logits = model(context)

        last_logits = logits[0].clone()

        # Reduce probability of recently used characters.
        for token_id in set(x[-context_size:].tolist()):

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
            torch.tensor(
                [next_id],
                dtype=torch.long
            )
        ])

        generated_text = tokenizer.decode(
            generated_ids
        )

        if _find_role_marker(generated_text) is not None:
            break

        # Stop when a blank paragraph appears.
        if "\n\n" in generated_text:
            break

    return clean_generated_text(tokenizer.decode(generated_ids))


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
