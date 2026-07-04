import unittest

from llm_guard.schema import SimpleJsonSchemaValidator


class SimpleJsonSchemaValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = SimpleJsonSchemaValidator()

    def test_valid_object_passes(self):
        schema = {
            "type": "object",
            "required": ["answer", "sources"],
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "sources": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "additionalProperties": False,
        }
        text = '{"answer":"Policy summary","sources":["https://example.com"]}'

        parsed, violations = self.validator.parse_and_validate(text, schema)

        self.assertEqual(parsed["answer"], "Policy summary")
        self.assertEqual(violations, [])

    def test_invalid_json_is_reported(self):
        _, violations = self.validator.parse_and_validate("{not-json}", {"type": "object"})

        self.assertEqual(len(violations), 1)
        self.assertIn("Invalid JSON", violations[0].message)

    def test_missing_required_field_is_reported(self):
        schema = {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        }

        _, violations = self.validator.parse_and_validate("{}", schema)

        self.assertEqual(len(violations), 1)
        self.assertIn("Missing required field", violations[0].message)


if __name__ == "__main__":
    unittest.main()
