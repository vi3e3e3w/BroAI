class CharTokenizer:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, text):
        chars = sorted(set(text))

        self.stoi = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        for i, ch in enumerate(chars, start=2):
            self.stoi[ch] = i

        self.itos = {
            i: ch
            for ch, i in self.stoi.items()
        }

    def encode(self, text):
        return [
            self.stoi.get(
                ch,
                self.stoi[self.UNK_TOKEN]
            )
            for ch in text
        ]

    def decode(self, ids):
        return "".join(
            self.itos.get(
                i,
                self.UNK_TOKEN
            )
            for i in ids
        )


if __name__ == "__main__":
    text = "Xin chao bro"

    tokenizer = CharTokenizer(text)

    print("Vocabulary size:", len(tokenizer.stoi))
    print("stoi:", tokenizer.stoi)

    encoded = tokenizer.encode("Xin chao bro")

    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))

    unknown = tokenizer.encode("Hello!")

    print("Unknown test:", unknown)
    print("Unknown decoded:", tokenizer.decode(unknown))
