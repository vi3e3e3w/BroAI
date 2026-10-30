import unittest

from src.learn import audit_chat, format_chat


class FormatChatTests(unittest.TestCase):
    def test_valid_legacy_and_tagged_roles_are_normalized(self):
        chat = "USER: Xin chao\nAI: Chao ban!\n<USER> Ban khoe khong?\n<AI> Minh khoe."
        self.assertEqual(
            format_chat(chat),
            "<USER> Xin chao\n<AI> Chao ban!\n<USER> Ban khoe khong?\n<AI> Minh khoe.",
        )

    def test_invalid_or_missing_role_markers_are_excluded(self):
        chat = "hello\nVISITOR: Xin chao\n<USER Xin chao\nUSER: Hop le"
        self.assertEqual(format_chat(chat), "<USER> Hop le")
        warnings = audit_chat(chat)
        self.assertTrue(any("unrecognized role marker" in warning for _, warning in warnings))

    def test_pair_order_and_incomplete_pairs_are_reported_not_filtered(self):
        chat = "<AI> Chao som\n<USER> Xin chao\n<USER> Cau tiep\n<AI> Tra loi"
        self.assertEqual(
            format_chat(chat),
            "<AI> Chao som\n<USER> Xin chao\n<USER> Cau tiep\n<AI> Tra loi",
        )
        warnings = audit_chat(chat)
        self.assertTrue(any("AI without" in warning for _, warning in warnings))
        self.assertTrue(any("USER has no following AI" in warning for _, warning in warnings))

    def test_invalid_line_breaks_pair_in_preview(self):
        warnings = audit_chat("<USER> Xin chao\nunknown line\n<AI> Chao")
        self.assertTrue(any("USER has no following AI" in warning for _, warning in warnings))
        self.assertTrue(any("AI without" in warning for _, warning in warnings))

    def test_ansi_and_disallowed_controls_are_removed(self):
        chat = "<USER> Xin\x1b[31m chao\x1b[0m\x07!\n<AI> \x1b]0;title\x07Chao\x0b ban"
        self.assertEqual(
            format_chat(chat),
            "<USER> Xin chao !\n<AI> Chao ban",
        )

    def test_vietnamese_text_and_punctuation_are_preserved(self):
        chat = "<USER> Xin chào, bạn khỏe không?\n<AI> Mình khỏe! Cảm ơn bạn."
        self.assertEqual(
            format_chat(chat),
            "<USER> Xin chào, bạn khỏe không?\n<AI> Mình khỏe! Cảm ơn bạn.",
        )


if __name__ == "__main__":
    unittest.main()
