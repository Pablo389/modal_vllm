"""Tests for structured JSON parsing."""

import unittest

from triton_bundle.client import StructuredParseError, parse_structured_bundle


class ParseStructuredTests(unittest.TestCase):
    def test_parses_clean_json(self) -> None:
        raw = '{"task": "div", "wrapper_name": "div"}'
        data = parse_structured_bundle(raw)
        self.assertEqual(data["task"], "div")

    def test_strips_fences(self) -> None:
        raw = '```json\n{"task": "x"}\n```'
        data = parse_structured_bundle(raw)
        self.assertEqual(data["task"], "x")

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(StructuredParseError):
            parse_structured_bundle('{"task": "div"')


if __name__ == "__main__":
    unittest.main()
