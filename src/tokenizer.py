class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))

        self.stoi = {
            ch: i
            for i, ch in enumerate(chars)
        }

        self.itos = {
            i: ch
            for ch, i in self.stoi.items()
        }

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    text = "Xin chào! Đây là AI."

    tokenizer = CharTokenizer(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Vocabulary:", tokenizer.stoi)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
