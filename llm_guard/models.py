from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    provider: str = "echo"
    model: str = "mock-model"
    messages: List[ChatMessage]
    response_schema: Optional[Dict[str, Any]] = None
    policy_profile: str = "default"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages must not be empty")
        return self


class GuardrailViolation(BaseModel):
    code: str
    phase: Literal["input", "output"]
    category: str
    message: str
    retryable: bool = False
    rule_id: Optional[str] = None


class GuardrailSummary(BaseModel):
    status: Literal["passed", "fallback"]
    policy_profile: str
    policy_version: str
    retry_count: int = 0
    input_violations: List[GuardrailViolation] = Field(default_factory=list)
    output_violations: List[GuardrailViolation] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    request_id: str
    provider: str
    model: str
    content: str
    structured_output: Optional[Any] = None
    finish_reason: Literal["stop", "fallback"] = "stop"
    guardrails: GuardrailSummary
    raw_provider_response: Optional[Dict[str, Any]] = None


class ProviderRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    content: str
    raw: Dict[str, Any] = Field(default_factory=dict)
