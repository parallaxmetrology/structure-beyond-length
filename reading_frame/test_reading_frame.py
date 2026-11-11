#!/usr/bin/env python3
"""Tests for reading frame r1. Run: python3 test_reading_frame.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_frame as RF

PLAIN = ("The bridge was finished in 1932. It carried trains for forty years, and then the "
         "city turned it into a footpath. People still argue about whether that was the right "
         "call, because the rail link was never replaced. Most residents walk it on weekends.")


class FrameRules(unittest.TestCase):
    def test_plain_text_is_identical(self):
        f = RF.frame_text(PLAIN)
        self.assertTrue(f.identical)
        self.assertEqual(f.text, PLAIN)
        self.assertFalse(f.edits.any)

    def test_bullets_untouched(self):
        t = "Steps:\n- Boil water\n- Add salt\n1. Stir\n2. Serve"
        self.assertTrue(RF.frame_text(t).identical)

    def test_code_block_removed(self):
        t = "Use this:\n```python\nprint('hi')\nx = 1\n```\nThat prints hi."
        f = RF.frame_text(t)
        self.assertNotIn("print('hi')", f.text)
        self.assertEqual(f.edits.code_blocks, 1)
        self.assertEqual(f.edits.code_lines, 4)
        self.assertIn("That prints hi.", f.text)

    def test_unclosed_fence_flagged(self):
        f = RF.frame_text("Here:\n```js\nfunction f() {")
        self.assertTrue(f.edits.unclosed_fence)
        self.assertNotIn("function", f.text)

    def test_header_dropped(self):
        f = RF.frame_text("## Overview\nThe plan has two parts.")
        self.assertEqual(f.text, "The plan has two parts.")

    def test_bold_line_dropped(self):
        f = RF.frame_text("**Ingredients:**\nFlour and water.")
        self.assertEqual(f.text, "Flour and water.")

    def test_header_between_paragraphs_keeps_break(self):
        f = RF.frame_text("First part.\n## Next\nSecond part.")
        self.assertEqual(f.text, "First part.\n\nSecond part.")

    def test_inline_bold_untouched(self):
        self.assertTrue(RF.frame_text("This is **very** important.").identical)

    def test_table_removed(self):
        t = "Compare:\n| a | b |\n|---|---|\n| 1 | 2 |\nDone."
        f = RF.frame_text(t)
        self.assertEqual(f.edits.table_lines, 3)
        self.assertNotIn("|", f.text)

    def test_link_keeps_text(self):
        f = RF.frame_text("See [the docs](https://example.com/a/b) for more.")
        self.assertEqual(f.text, "See the docs for more.")

    def test_language_gate(self):
        self.assertIn("non_latin_script", RF.frame_text("抱歉，您给出的句子似乎是混乱的字符和单词，我无法理解其含义。").reasons)
        self.assertIn("no_letters", RF.frame_text("1120 = 4 * 280").reasons)
        es = ("El puente fue terminado en mil novecientos treinta y dos y llevó trenes durante "
              "cuarenta años hasta que la ciudad lo convirtió en un paseo peatonal muy popular")
        self.assertIn("low_english_function_words", RF.frame_text(es).reasons)
        self.assertTrue(RF.frame_text(PLAIN).english)

    def test_short_gate(self):
        f = RF.frame_text("Hello! How can I help you today?")
        self.assertTrue(f.english)
        self.assertFalse(f.eligible)
        self.assertEqual(f.reasons, ["short_prose"])

    def test_collapse_reading(self):
        self.assertEqual(RF.collapse_reading({"collapse_subtype": "severe", "tail_incomplete_flag": False}), "short_complete")
        self.assertEqual(RF.collapse_reading({"collapse_subtype": "severe", "tail_incomplete_flag": True},
                                             "Sure, here are the steps to"), "short_cut")
        self.assertEqual(RF.collapse_reading({"collapse_subtype": "severe", "tail_incomplete_flag": True},
                                             "I can't help with that."), "short_complete")
        self.assertEqual(RF.collapse_reading({"collapse_subtype": "severe", "tail_incomplete_flag": True},
                                             "Here are some steps on how to grow potatoes:"), "short_cut")
        self.assertEqual(RF.collapse_reading({"collapse_subtype": "math"}), "cut_code_or_math")
        self.assertEqual(RF.collapse_reading({"collapse_subtype": "partial"}), "cut_partial")


class WithKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = RF.Reader()

    def test_hash_is_frozen_v19(self):
        self.assertEqual(self.R.kernel_sha256, RF.V19_SHA256)

    def test_plain_raw_equals_framed_equals_direct(self):
        r = self.R.read(PLAIN)
        self.assertEqual(r["raw"], r["framed"])
        self.assertTrue(all(v == 0 for v in r["delta"].values()))
        direct = self.R.kernel.compute(PLAIN)
        for d in RF.DIMS:
            self.assertEqual(r["raw"]["coords"][d], {**direct["kernel"], **direct["aux"]}[d])

    def test_header_glue_is_reduced(self):
        t = ("## Background\nThe survey ran for two years. Results were mixed.\n\n"
             "## Findings\nMost sites improved. A few did not.")
        r = self.R.read(t)
        self.assertLess(r["framed"]["coords"]["Δy"], r["raw"]["coords"]["Δy"])

    def test_short_refusal(self):
        r = self.R.read("I'm sorry, but I can't help with that.")
        self.assertEqual(r["raw"]["collapse_subtype"], "severe")
        self.assertEqual(r["collapse_reading"], "short_complete")

    def test_flat_row(self):
        row = RF.flat_row(self.R.read(PLAIN))
        self.assertIn("raw_Δy", row)
        self.assertIn("framed_μ", row)


if __name__ == "__main__":
    unittest.main(verbosity=1)
