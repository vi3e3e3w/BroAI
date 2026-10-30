from pathlib import Path
import re
import unicodedata


RAW_PATH = Path("data/raw/train.txt")
CHAT_PATH = Path("data/conversations/chat.txt")
OUTPUT_PATH = Path("data/processed/train.txt")

USER_TAG = "<USER>"
AI_TAG = "<AI>"
ROLE_PREFIXES = ("USER:", "AI:", USER_TAG, AI_TAG)
ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\)|[@-Z\\-_])"
)


def _clean_content(text):
    text = ANSI_ESCAPE_RE.sub("", text)
    cleaned = []
    for index, char in enumerate(text):
        if unicodedata.category(char) == "Cc":
            if (
                (cleaned and cleaned[-1].isspace())
                or (index + 1 < len(text) and text[index + 1].isspace())
            ):
                continue
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return "".join(cleaned).strip()


def _parse_role_line(line):
    for prefix in ROLE_PREFIXES:
        if line.startswith(prefix):
            role = "USER" if prefix in ("USER:", USER_TAG) else "AI"
            content = _clean_content(line[len(prefix):])
            return role, content
    return None


def audit_chat(chat_data):
    """Return line-numbered warnings without excluding role-order errors."""
    warnings = []
    expected_role = "USER"
    pending_user_line = None

    for line_number, raw_line in enumerate(chat_data.split("\n"), start=1):
        line = _clean_content(raw_line)
        if ANSI_ESCAPE_RE.search(raw_line) or any(
            unicodedata.category(char) == "Cc" for char in raw_line
        ):
            warnings.append((line_number, "ANSI/control characters removed"))
        if not line:
            continue

        parsed = _parse_role_line(line)
        if parsed is None:
            warnings.append((line_number, "unrecognized role marker; excluded"))
            if pending_user_line is not None:
                warnings.append((pending_user_line, "USER has no following AI line"))
            expected_role = "USER"
            pending_user_line = None
            continue

        role, content = parsed
        if not content:
            warnings.append((line_number, "empty content after control cleanup; excluded"))
            if pending_user_line is not None:
                warnings.append((pending_user_line, "USER has no following AI line"))
            expected_role = "USER"
            pending_user_line = None
            continue

        if role != expected_role:
            if role == "USER":
                warnings.append((pending_user_line, "USER has no following AI line"))
                pending_user_line = line_number
                expected_role = "AI"
            else:
                warnings.append((line_number, "AI without a preceding USER line"))
            continue

        if role == "USER":
            pending_user_line = line_number
            expected_role = "AI"
        else:
            pending_user_line = None
            expected_role = "USER"

    if expected_role == "AI" and pending_user_line is not None:
        warnings.append((pending_user_line, "USER has no following AI line"))

    return warnings


def format_chat(chat_data):
    lines = []

    for line in chat_data.split("\n"):
        line = _clean_content(line)

        if not line:
            continue

        parsed = _parse_role_line(line)
        if parsed is None:
            continue

        role, content = parsed
        role_tag = USER_TAG if role == "USER" else AI_TAG

        if content:
            lines.append(f"{role_tag} {content}")

    return "\n".join(lines)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_data = ""

    if RAW_PATH.exists():
        raw_data = RAW_PATH.read_text(encoding="utf-8")

    chat_data = ""

    if CHAT_PATH.exists():
        chat_data = CHAT_PATH.read_text(encoding="utf-8")

    for line_number, warning in audit_chat(chat_data):
        print(f"Chat data warning (line {line_number}): {warning}")

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
