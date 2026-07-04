import unittest

from llm_guard.policy import PolicyEngine


class PolicyEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PolicyEngine.from_dict(
            {
                "version": "test",
                "profiles": {
                    "default": {
                        "rules": [
                            {
                                "id": "block-competitor",
                                "phase": "output",
                                "action": "block",
                                "message": "Competitor discussion is not allowed.",
                                "match": {"type": "keyword", "keywords": ["Anthropic"]},
                            },
                            {
                                "id": "redact-secret",
                                "phase": "input",
                                "action": "redact",
                                "message": "Redact the internal codename.",
                                "match": {"type": "keyword", "keywords": ["Project Atlas"]},
                            },
                        ]
                    }
                },
            }
        )
        self.profile = self.engine.get_profile("default")

    def test_keyword_rule_matches(self):
        matches = self.engine.evaluate("Do not compare us to Anthropic.", self.profile, "output")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].rule.id, "block-competitor")

    def test_redaction_rewrites_matching_text(self):
        rule = self.engine.get_rules(self.profile, "input")[0]
        text = self.engine.redact_text("Project Atlas is the codename.", rule)

        self.assertEqual(text, "[REDACTED] is the codename.")


if __name__ == "__main__":
    unittest.main()
