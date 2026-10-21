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

    @classmethod
    def from_mappings(cls, stoi, itos=None):
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = {
            token: int(index)
            for token, index in stoi.items()
        }

        if itos is None:
            tokenizer.itos = {
                index: token
                for token, index in tokenizer.stoi.items()
            }
        else:
            tokenizer.itos = {
                int(index): token
                for index, token in itos.items()
            }

        expected_ids = set(range(len(tokenizer.stoi)))
        if (
            tokenizer.stoi.get(cls.PAD_TOKEN) != 0
            or tokenizer.stoi.get(cls.UNK_TOKEN) != 1
            or set(tokenizer.stoi.values()) != expected_ids
            or len(tokenizer.itos) != len(tokenizer.stoi)
            or any(
                tokenizer.itos.get(index) != token
                for token, index in tokenizer.stoi.items()
            )
        ):
            raise ValueError("Invalid tokenizer mappings in checkpoint")

        return tokenizer

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
