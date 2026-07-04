from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from llm_guard.detectors import CitationDetector, InjectionDetector, PIIDetector, TopicDetector, ToxicityDetector
from llm_guard.errors import GuardrailError, PolicyError
from llm_guard.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    GuardrailSummary,
    GuardrailViolation,
    ProviderRequest,
)
from llm_guard.policy import PolicyEngine, PolicyProfile, RuleConfig
from llm_guard.providers.base import LLMProvider
from llm_guard.schema import SimpleJsonSchemaValidator


@dataclass
class OutputValidationResult:
    ok: bool
    text: str
    structured_output: Optional[Any]
    violations: List[GuardrailViolation]
    use_fallback: bool = False


class GuardrailService:
    def __init__(
        self,
        policy_engine: PolicyEngine,
        providers: Dict[str, LLMProvider],
        schema_validator: Optional[SimpleJsonSchemaValidator] = None,
        injection_detector: Optional[InjectionDetector] = None,
        pii_detector: Optional[PIIDetector] = None,
        toxicity_detector: Optional[ToxicityDetector] = None,
        topic_detector: Optional[TopicDetector] = None,
        citation_detector: Optional[CitationDetector] = None,
    ) -> None:
        self.policy_engine = policy_engine
        self.providers = providers
        self.schema_validator = schema_validator or SimpleJsonSchemaValidator()
        self.injection_detector = injection_detector or InjectionDetector()
        self.pii_detector = pii_detector or PIIDetector()
        self.toxicity_detector = toxicity_detector or ToxicityDetector()
        self.topic_detector = topic_detector or TopicDetector()
        self.citation_detector = citation_detector or CitationDetector()

    def handle_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        request_id = str(uuid4())
        profile = self._load_profile(request.policy_profile)
        provider = self._resolve_provider(request.provider, profile)
        prompt_preview = self._redacted_preview(request.messages, profile)

        transformed_messages, input_violations = self._validate_and_transform_input(request, profile)
        if input_violations:
            raise GuardrailError(
                http_status=422,
                error_code="GUARDRAIL_INPUT_BLOCKED",
                message="The request was blocked by input guardrails.",
                violations=input_violations,
                details={"request_id": request_id, "prompt_preview": prompt_preview},
            )

        retries = 0
        current_messages = transformed_messages
        last_violations: List[GuardrailViolation] = []

        while True:
            provider_request = ProviderRequest(
                model=request.model,
                messages=current_messages,
                parameters=request.parameters,
                metadata=request.metadata,
            )
            provider_response = self._invoke_provider(provider, provider_request)
            output_result = self._validate_output(
                content=provider_response.content,
                request=request,
                profile=profile,
            )
            last_violations = output_result.violations

            if output_result.ok:
                return ChatCompletionResponse(
                    request_id=request_id,
                    provider=request.provider,
                    model=request.model,
                    content=output_result.text,
                    structured_output=output_result.structured_output,
                    finish_reason="stop",
                    guardrails=GuardrailSummary(
                        status="passed",
                        policy_profile=request.policy_profile,
                        policy_version=self.policy_engine.config.version,
                        retry_count=retries,
                    ),
                    raw_provider_response=provider_response.raw,
                )

            if retries < profile.retry.max_attempts and self._can_retry(output_result.violations) and not output_result.use_fallback:
                retries += 1
                current_messages = self._build_repair_messages(
                    original_messages=transformed_messages,
                    previous_output=provider_response.content,
                    violations=output_result.violations,
                    response_schema=request.response_schema,
                )
                continue

            fallback_response = self._build_fallback_response(
                request=request,
                profile=profile,
                request_id=request_id,
                retries=retries,
                violations=last_violations,
            )
            if fallback_response is not None:
                return fallback_response

            raise GuardrailError(
                http_status=422,
                error_code="GUARDRAIL_OUTPUT_UNAVAILABLE",
                message="The model could not produce a safe, policy-compliant response.",
                violations=last_violations,
                details={"request_id": request_id},
            )

    def readiness(self) -> Dict[str, Any]:
        try:
            self._load_profile("default")
            return {"status": "ready", "providers": sorted(self.providers.keys())}
        except PolicyError as exc:
            raise GuardrailError(500, "POLICY_INVALID", str(exc))

    def _validate_and_transform_input(
        self,
        request: ChatCompletionRequest,
        profile: PolicyProfile,
    ) -> (List[ChatMessage], List[GuardrailViolation]):
        messages = [message.model_copy(deep=True) for message in request.messages]
        prompt_text = self._join_messages(messages)
        violations: List[GuardrailViolation] = []

        if len(prompt_text) > profile.input.max_chars:
            violations.append(
                GuardrailViolation(
                    code="INPUT_TOO_LARGE",
                    phase="input",
                    category="size",
                    message="The request exceeds the configured input size limit.",
                )
            )

        if profile.input.block_prompt_injection:
            injection_hits = self.injection_detector.detect(prompt_text)
            if injection_hits:
                violations.append(
                    GuardrailViolation(
                        code="PROMPT_INJECTION_DETECTED",
                        phase="input",
                        category="prompt_injection",
                        message="Prompt injection or jailbreak instructions were detected.",
                    )
                )

        if profile.input.block_pii:
            pii_hits = self.pii_detector.detect(prompt_text)
            if pii_hits:
                violations.append(
                    GuardrailViolation(
                        code="PII_DETECTED",
                        phase="input",
                        category="pii",
                        message="Sensitive information was detected in the request.",
                    )
                )

        if violations:
            return messages, violations

        rule_matches = self.policy_engine.evaluate(prompt_text, profile, phase="input")
        rewrite_messages: List[ChatMessage] = []
        for match in rule_matches:
            rule = match.rule
            if rule.action == "block" or rule.action == "fallback":
                violations.append(self._rule_violation(rule, "input"))
            elif rule.action == "redact":
                messages = [message.model_copy(update={"content": self.policy_engine.redact_text(message.content, rule)}) for message in messages]
            elif rule.action == "rewrite":
                rewrite_messages.append(
                    ChatMessage(
                        role="system",
                        content=rule.rewrite_instruction or rule.message,
                    )
                )

        if violations:
            return messages, violations

        if rewrite_messages:
            messages = rewrite_messages + messages

        return messages, []

    def _validate_output(
        self,
        content: str,
        request: ChatCompletionRequest,
        profile: PolicyProfile,
    ) -> OutputValidationResult:
        text = content.strip()
        prompt_text = self._join_messages(request.messages)
        topic_score = self.topic_detector.score(prompt_text, text)
        violations: List[GuardrailViolation] = []
        structured_output: Optional[Any] = None
        use_fallback = False

        rule_matches = self.policy_engine.evaluate(text, profile, phase="output", context={"topic_score": topic_score})
        for match in rule_matches:
            rule = match.rule
            if rule.action == "redact":
                text = self.policy_engine.redact_text(text, rule)
            elif rule.action == "fallback":
                violations.append(self._rule_violation(rule, "output"))
                use_fallback = True
            elif rule.action == "require_citation":
                pass
            elif rule.action == "require_topic":
                pass
            elif rule.action == "require_json":
                pass
            else:
                violations.append(self._rule_violation(rule, "output"))

        if len(text) > profile.output.max_chars:
            violations.append(
                GuardrailViolation(
                    code="OUTPUT_TOO_LARGE",
                    phase="output",
                    category="size",
                    message="The model response exceeds the configured output size limit.",
                    retryable=True,
                )
            )

        if profile.output.block_toxicity:
            toxic_hits = self.toxicity_detector.detect(text)
            if toxic_hits:
                violations.append(
                    GuardrailViolation(
                        code="TOXIC_CONTENT_DETECTED",
                        phase="output",
                        category="toxicity",
                        message="The model response contains toxic or abusive language.",
                        retryable=True,
                    )
                )

        if profile.output.block_pii:
            pii_hits = self.pii_detector.detect(text)
            if pii_hits:
                violations.append(
                    GuardrailViolation(
                        code="OUTPUT_PII_DETECTED",
                        phase="output",
                        category="pii",
                        message="Sensitive information was detected in the model response.",
                        retryable=True,
                    )
                )

        requires_json = request.response_schema is not None or profile.output.require_json_by_default or self._has_rule_action(rule_matches, "require_json")
        if requires_json:
            structured_output, schema_violations = self.schema_validator.parse_and_validate(text, request.response_schema)
            if schema_violations:
                violations.append(
                    GuardrailViolation(
                        code="SCHEMA_VALIDATION_FAILED",
                        phase="output",
                        category="schema",
                        message="Response schema validation failed: {0}".format("; ".join(item.message for item in schema_violations)),
                        retryable=True,
                    )
                )

        requires_citations = profile.output.require_citations or self._has_rule_action(rule_matches, "require_citation")
        if requires_citations and not self.citation_detector.has_citation(text, structured_output):
            violations.append(
                GuardrailViolation(
                    code="CITATION_REQUIRED",
                    phase="output",
                    category="citation",
                    message="The response must include a citation or sources field.",
                    retryable=True,
                )
            )

        requires_topic = self._has_rule_action(rule_matches, "require_topic") or profile.output.topic_threshold > 0
        if requires_topic and topic_score < profile.output.topic_threshold:
            violations.append(
                GuardrailViolation(
                    code="OFF_TOPIC_RESPONSE",
                    phase="output",
                    category="topic",
                    message="The response does not stay sufficiently on topic.",
                    retryable=True,
                )
            )

        return OutputValidationResult(
            ok=not violations,
            text=text,
            structured_output=structured_output,
            violations=violations,
            use_fallback=use_fallback,
        )

    def _build_repair_messages(
        self,
        original_messages: List[ChatMessage],
        previous_output: str,
        violations: List[GuardrailViolation],
        response_schema: Optional[Dict[str, Any]],
    ) -> List[ChatMessage]:
        bullet_lines = ["- {0}".format(violation.message) for violation in violations]
        instructions = [
            "Your previous answer failed middleware validation.",
            "Regenerate the answer only and fix these issues:",
            "\n".join(bullet_lines),
            "Do not mention the validation process.",
        ]
        if response_schema is not None:
            instructions.append("Return valid JSON only that matches this schema exactly:")
            instructions.append(json.dumps(response_schema, ensure_ascii=True))

        repair_message = ChatMessage(role="system", content="\n".join(instructions))
        prior_answer = ChatMessage(role="assistant", content=previous_output)
        return [repair_message] + [message.model_copy(deep=True) for message in original_messages] + [prior_answer]

    def _build_fallback_response(
        self,
        request: ChatCompletionRequest,
        profile: PolicyProfile,
        request_id: str,
        retries: int,
        violations: List[GuardrailViolation],
    ) -> Optional[ChatCompletionResponse]:
        if request.response_schema is not None:
            if profile.fallback_json is None:
                return None
            schema_violations = self.schema_validator.validate_instance(profile.fallback_json, request.response_schema)
            if schema_violations:
                return None
            content = json.dumps(profile.fallback_json)
            structured_output = profile.fallback_json
        else:
            content = profile.fallback_message
            structured_output = None

        return ChatCompletionResponse(
            request_id=request_id,
            provider=request.provider,
            model=request.model,
            content=content,
            structured_output=structured_output,
            finish_reason="fallback",
            guardrails=GuardrailSummary(
                status="fallback",
                policy_profile=request.policy_profile,
                policy_version=self.policy_engine.config.version,
                retry_count=retries,
                output_violations=violations,
            ),
        )

    def _rule_violation(self, rule: RuleConfig, phase: str) -> GuardrailViolation:
        return GuardrailViolation(
            code="POLICY_RULE_VIOLATION",
            phase=phase,
            category="policy",
            message=rule.message,
            retryable=self.policy_engine.default_retryable(rule),
            rule_id=rule.id,
        )

    def _resolve_provider(self, provider_name: str, profile: PolicyProfile) -> LLMProvider:
        if profile.providers_allow is not None and provider_name not in profile.providers_allow:
            raise GuardrailError(
                http_status=422,
                error_code="GUARDRAIL_INPUT_BLOCKED",
                message="The requested provider is not allowed by policy.",
                violations=[
                    GuardrailViolation(
                        code="PROVIDER_NOT_ALLOWED",
                        phase="input",
                        category="policy",
                        message="Provider '{0}' is not allowed for this policy profile.".format(provider_name),
                    )
                ],
            )
        try:
            return self.providers[provider_name]
        except KeyError as exc:
            raise GuardrailError(
                http_status=502,
                error_code="UPSTREAM_LLM_ERROR",
                message="No provider adapter is configured for '{0}'.".format(provider_name),
            ) from exc

    def _invoke_provider(self, provider: LLMProvider, provider_request: ProviderRequest):
        try:
            return provider.complete(provider_request)
        except GuardrailError:
            raise
        except Exception as exc:
            raise GuardrailError(
                http_status=502,
                error_code="UPSTREAM_LLM_ERROR",
                message="The upstream model provider failed.",
                details={"reason": str(exc)},
            ) from exc

    def _load_profile(self, profile_name: str) -> PolicyProfile:
        try:
            return self.policy_engine.get_profile(profile_name)
        except PolicyError as exc:
            raise GuardrailError(400, "POLICY_INVALID", str(exc)) from exc

    def _can_retry(self, violations: List[GuardrailViolation]) -> bool:
        return any(violation.retryable for violation in violations)

    def _has_rule_action(self, matches, action: str) -> bool:
        return any(match.rule.action == action for match in matches)

    def _join_messages(self, messages: List[ChatMessage]) -> str:
        return "\n".join(message.content for message in messages if message.content)

    def _redacted_preview(self, messages: List[ChatMessage], profile: PolicyProfile) -> str:
        preview = self._join_messages(messages)
        if profile.input.redact_logs:
            return self.pii_detector.redact(preview)
        return preview
