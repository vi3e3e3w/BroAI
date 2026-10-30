from src.generate import generate


MAX_NEW_TOKENS = 100
CONVERSATION_PATH = "data/conversations/chat.txt"
USER_TAG = "<USER>"
AI_TAG = "<AI>"


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

        if prompt.lower() == "/exit":
            print("AI: Bye!")
            break

        if not prompt:
            continue

        try:
            tagged_prompt = f"{USER_TAG} {prompt}\n{AI_TAG} "

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
                clean_prompt = prompt.replace("\r", " ").replace("\n", " ")
                clean_response = response.replace("\r", " ").replace("\n", " ")
                f.write(f"{USER_TAG} {clean_prompt}\n")
                f.write(f"{AI_TAG} {clean_response}\n")
                f.write("\n")

        except KeyError as e:
            print(
                f"AI: I don't know this character yet: {e}"
            )


if __name__ == "__main__":
     main()
