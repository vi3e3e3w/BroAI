import unittest
from unittest.mock import patch

import torch

import src.generate as generation
from src.generate import (
    _find_role_marker,
    checkpoint,
    clean_generated_text,
    context_size,
    tokenizer,
)
from src.tokenizer import CharTokenizer


class GeneratedTagCleanupTests(unittest.TestCase):
    def generate_from_fixed_characters(self, characters):
        generated = iter(characters)
        generated_characters = []

        def fixed_logits(context):
            char = next(generated)
            generated_characters.append(char)
            logits = torch.zeros((context.shape[0], len(tokenizer.stoi)))
            logits[:, tokenizer.stoi[char]] = 100
            return logits

        with (
            patch.object(generation.model, "forward", side_effect=fixed_logits),
            patch("torch.multinomial", return_value=torch.tensor([0])),
        ):
            result = generation.generate("prompt", max_new_tokens=40)

        return result, "".join(generated_characters)

    def test_normal_vietnamese_and_punctuation_are_preserved(self):
        text = "Xin chao bro! Hom nay ban the nao?"
        self.assertEqual(clean_generated_text(text), text)

    def test_complete_role_tags_stop_visible_output(self):
        self.assertEqual(clean_generated_text("<AI> hidden"), "")
        self.assertEqual(clean_generated_text("Xin chao<USER> hidden"), "Xin chao")
        self.assertEqual(clean_generated_text("Xin chao<AI>"), "Xin chao")

    def test_standalone_angle_brackets_are_preserved(self):
        for text in ("<", "2 < 3", "x << 4", "xin chao < bro"):
            with self.subTest(text=text):
                self.assertEqual(clean_generated_text(text), text)

    def test_incomplete_role_prefixes_are_removed(self):
        cases = {
            "<US": "",
            "<AI": "",
            "Xin <US": "Xin",
            "Xin <AI noi tiep": "Xin noi tiep",
            "Xin USER> noi tiep": "Xin noi tiep",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(clean_generated_text(text), expected)

    def test_malformed_ar_marker_fragments_are_contextual(self):
        self.assertEqual(clean_generated_text("AR> noi tiep"), "noi tiep")
        self.assertEqual(clean_generated_text("Xin chao. AR> noi tiep"), "Xin chao. noi tiep")
        self.assertEqual(clean_generated_text("<AR> noi tiep"), "noi tiep")
        self.assertEqual(clean_generated_text("Xin chao <AR"), "Xin chao")
        self.assertEqual(clean_generated_text("Xin chao <AR> noi tiep"), "Xin chao noi tiep")

        for text in (
            "CAR> is text",
            "xAR>y",
            "AR>VR",
            "The AR> marker",
            "x<AR>y",
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_generated_text(text), text)

    def test_multiline_response_stops_at_next_role_tag(self):
        text = "Xin chao.\nToi khoe.\n<USER> next turn"
        self.assertEqual(clean_generated_text(text), "Xin chao.\nToi khoe.")

    def test_generation_waits_for_ambiguous_tag_prefix(self):
        self.assertIsNone(_find_role_marker("Xin <"))
        self.assertIsNone(_find_role_marker("Xin <US"))
        self.assertIsNotNone(_find_role_marker("Xin <US next"))
        self.assertIsNotNone(_find_role_marker("Xin <AI> next"))
        self.assertIsNotNone(_find_role_marker("Xin. AR> next"))

    def test_generation_stops_at_next_complete_role_tag(self):
        result, generated = self.generate_from_fixed_characters("Xin chao<AI>hidden")
        self.assertEqual(result, "Xin chao")
        self.assertEqual(generated, "Xin chao<AI>")

    def test_generation_stops_at_observed_ar_fragment(self):
        result, generated = self.generate_from_fixed_characters(
            "A sentence. AR>hidden"
        )
        self.assertEqual(result, "A sentence.")
        self.assertEqual(generated, "A sentence. AR>")

    def test_checkpoint_tokenizer_mapping_is_used_unchanged(self):
        restored = CharTokenizer.from_mappings(
            checkpoint["stoi"],
            checkpoint["itos"]
        )
        self.assertEqual(tokenizer.stoi, checkpoint["stoi"])
        self.assertEqual(restored.stoi, checkpoint["stoi"])
        self.assertEqual(restored.itos, checkpoint["itos"])
        self.assertEqual(context_size, checkpoint["context_size"])


if __name__ == "__main__":
    unittest.main()
