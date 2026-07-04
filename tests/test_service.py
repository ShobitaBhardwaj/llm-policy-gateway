import unittest

from llm_guard.errors import GuardrailError
from llm_guard.models import ChatCompletionRequest, ProviderRequest, ProviderResponse
from llm_guard.policy import PolicyEngine
from llm_guard.providers.base import LLMProvider
from llm_guard.service import GuardrailService


class SequenceProvider(LLMProvider):
    name = "test"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self.calls >= len(self.responses):
            content = self.responses[-1]
        else:
            content = self.responses[self.calls]
        self.calls += 1
        return ProviderResponse(content=content, raw={"call": self.calls})


def build_policy(require_citations=True, retry_attempts=1):
    return PolicyEngine.from_dict(
        {
            "version": "test",
            "profiles": {
                "default": {
                    "providers_allow": ["test"],
                    "input": {
                        "max_chars": 5000,
                        "block_prompt_injection": True,
                        "block_pii": True,
                    },
                    "output": {
                        "max_chars": 5000,
                        "block_toxicity": True,
                        "block_pii": True,
                        "topic_threshold": 0.0,
                        "require_citations": require_citations,
                        "require_json_by_default": False,
                    },
                    "retry": {"max_attempts": retry_attempts},
                    "fallback_message": "Safe fallback.",
                    "rules": [
                        {
                            "id": "block-medical-advice",
                            "phase": "output",
                            "action": "block",
                            "retryable": False,
                            "message": "Medical advice is not allowed.",
                            "match": {"type": "keyword", "keywords": ["dosage"]},
                        }
                    ],
                }
            },
        }
    )


class GuardrailServiceTests(unittest.TestCase):
    def test_blocks_sensitive_input_before_provider(self):
        service = GuardrailService(build_policy(), {"test": SequenceProvider(['{"answer":"ok","sources":["https://example.com"]}'])})
        request = ChatCompletionRequest(
            provider="test",
            model="mock",
            messages=[{"role": "user", "content": "My card is 4111 1111 1111 1111"}],
        )

        with self.assertRaises(GuardrailError) as error:
            service.handle_completion(request)

        self.assertEqual(error.exception.error_code, "GUARDRAIL_INPUT_BLOCKED")
        self.assertEqual(error.exception.violations[0].code, "PII_DETECTED")

    def test_retries_invalid_output_and_returns_structured_response(self):
        provider = SequenceProvider(
            [
                '{"answer":"Refunds are supported."}',
                '{"answer":"Refunds are supported.","sources":["https://example.com/policy"]}',
            ]
        )
        service = GuardrailService(build_policy(), {"test": provider})
        request = ChatCompletionRequest(
            provider="test",
            model="mock",
            messages=[{"role": "user", "content": "Return refund guidance as JSON with sources."}],
            response_schema={
                "type": "object",
                "required": ["answer", "sources"],
                "properties": {
                    "answer": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "additionalProperties": False,
            },
        )

        response = service.handle_completion(request)

        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.guardrails.retry_count, 1)
        self.assertEqual(response.structured_output["sources"][0], "https://example.com/policy")

    def test_returns_fallback_for_non_retryable_policy_violation(self):
        provider = SequenceProvider(["You should change the dosage immediately. [1]"])
        service = GuardrailService(build_policy(require_citations=False, retry_attempts=0), {"test": provider})
        request = ChatCompletionRequest(
            provider="test",
            model="mock",
            messages=[{"role": "user", "content": "Give me a product summary."}],
        )

        response = service.handle_completion(request)

        self.assertEqual(response.finish_reason, "fallback")
        self.assertEqual(response.content, "Safe fallback.")
        self.assertEqual(response.guardrails.output_violations[0].rule_id, "block-medical-advice")

    def test_schema_failure_without_fallback_json_raises_error(self):
        provider = SequenceProvider(["not json", "still not json"])
        service = GuardrailService(build_policy(require_citations=False, retry_attempts=1), {"test": provider})
        request = ChatCompletionRequest(
            provider="test",
            model="mock",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_schema={
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "string"}},
            },
        )

        with self.assertRaises(GuardrailError) as error:
            service.handle_completion(request)

        self.assertEqual(error.exception.error_code, "GUARDRAIL_OUTPUT_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
