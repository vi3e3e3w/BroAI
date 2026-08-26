from pathlib import Path


RAW_PATH = Path("data/raw/train.txt")
CHAT_PATH = Path("data/conversations/chat.txt")
OUTPUT_PATH = Path("data/processed/train.txt")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_data = ""

    if RAW_PATH.exists():
        raw_data = RAW_PATH.read_text(encoding="utf-8")

    chat_data = ""

    if CHAT_PATH.exists():
        chat_data = CHAT_PATH.read_text(encoding="utf-8")

    combined = raw_data.rstrip()

    if chat_data.strip():
        combined += "\n\n" + chat_data.strip()

    OUTPUT_PATH.write_text(
        combined + "\n",
        encoding="utf-8"
    )

    print("Learning dataset updated!")
    print("Saved:", OUTPUT_PATH)
    print("Characters:", len(combined))


if __name__ == "__main__":
    main()
