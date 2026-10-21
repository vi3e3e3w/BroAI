from pathlib import Path


RAW_PATH = Path("data/raw/train.txt")
CHAT_PATH = Path("data/conversations/chat.txt")
OUTPUT_PATH = Path("data/processed/train.txt")

USER_TAG = "<USER>"
AI_TAG = "<AI>"


def format_chat(chat_data):
    lines = []

    for line in chat_data.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("USER:"):
            content = line[len("USER:"):].strip()
            lines.append(f"{USER_TAG} {content}")

        elif line.startswith("AI:"):
            content = line[len("AI:"):].strip()
            lines.append(f"{AI_TAG} {content}")

        else:
            # Giữ nguyên những dòng chưa có role tag
            lines.append(line)

    return "\n".join(lines)


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
        formatted_chat = format_chat(chat_data)

        if combined:
            combined += "\n\n"

        combined += formatted_chat

    OUTPUT_PATH.write_text(
        combined + "\n",
        encoding="utf-8"
    )

    print("Learning dataset updated!")
    print("Saved:", OUTPUT_PATH)
    print("Characters:", len(combined))


if __name__ == "__main__":
    main()
