from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from llm_guard.errors import PolicyError


class InputPolicy(BaseModel):
    max_chars: int = 16000
    redact_logs: bool = True
    block_prompt_injection: bool = True
    block_pii: bool = True


class OutputPolicy(BaseModel):
    max_chars: int = 12000
    block_toxicity: bool = True
    block_pii: bool = True
    topic_threshold: float = 0.10
    require_citations: bool = False
    require_json_by_default: bool = False


class RetryPolicy(BaseModel):
    max_attempts: int = 1


class MatchConfig(BaseModel):
    type: Literal["always", "keyword", "regex", "topic"] = "always"
    keywords: List[str] = Field(default_factory=list)
    pattern: Optional[str] = None
    min_score: Optional[float] = None
    case_sensitive: bool = False


class RuleConfig(BaseModel):
    id: str
    phase: Literal["input", "output"]
    action: Literal["block", "redact", "rewrite", "require_citation", "require_topic", "require_json", "fallback"]
    message: str
    match: MatchConfig = Field(default_factory=MatchConfig)
    enabled: bool = True
    priority: int = 100
    retryable: Optional[bool] = None
    rewrite_instruction: Optional[str] = None


class PolicyProfile(BaseModel):
    providers_allow: Optional[List[str]] = None
    input: InputPolicy = Field(default_factory=InputPolicy)
    output: OutputPolicy = Field(default_factory=OutputPolicy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    fallback_message: str = "I can’t provide a safe answer for that request."
    fallback_json: Optional[Any] = None
    rules: List[RuleConfig] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    version: str = "1"
    profiles: Dict[str, PolicyProfile]


@dataclass(frozen=True)
class RuleMatch:
    rule: RuleConfig
    matched_values: List[str]


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    @classmethod
    def from_file(cls, path: str) -> "PolicyEngine":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            raise PolicyError("Policy file not found: {0}".format(path)) from exc
        except yaml.YAMLError as exc:
            raise PolicyError("Policy file is invalid YAML: {0}".format(exc)) from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyEngine":
        try:
            config = PolicyConfig.model_validate(data)
        except Exception as exc:
            raise PolicyError("Policy configuration is invalid: {0}".format(exc)) from exc
        return cls(config)

    def get_profile(self, profile_name: str) -> PolicyProfile:
        try:
            return self.config.profiles[profile_name]
        except KeyError as exc:
            raise PolicyError("Unknown policy profile: {0}".format(profile_name)) from exc

    def get_rules(self, profile: PolicyProfile, phase: str) -> List[RuleConfig]:
        rules = [rule for rule in profile.rules if rule.phase == phase and rule.enabled]
        return sorted(rules, key=lambda rule: rule.priority)

    def evaluate(self, text: str, profile: PolicyProfile, phase: str, context: Optional[Dict[str, Any]] = None) -> List[RuleMatch]:
        context = context or {}
        matches: List[RuleMatch] = []
        for rule in self.get_rules(profile, phase):
            values = self._matched_values(rule, text, context)
            if values:
                matches.append(RuleMatch(rule=rule, matched_values=values))
        return matches

    def redact_text(self, text: str, rule: RuleConfig) -> str:
        if rule.match.type == "keyword":
            updated = text
            for keyword in rule.match.keywords:
                updated = re.sub(re.escape(keyword), "[REDACTED]", updated, flags=self._re_flags(rule.match))
            return updated
        if rule.match.type == "regex" and rule.match.pattern:
            return re.sub(rule.match.pattern, "[REDACTED]", text, flags=self._re_flags(rule.match))
        return text

    def default_retryable(self, rule: RuleConfig) -> bool:
        if rule.retryable is not None:
            return rule.retryable
        if rule.phase == "input":
            return False
        return rule.action not in ("fallback",)

    def _matched_values(self, rule: RuleConfig, text: str, context: Dict[str, Any]) -> List[str]:
        match_type = rule.match.type
        if match_type == "always":
            return ["always"]
        if match_type == "keyword":
            found = []
            haystack = text if rule.match.case_sensitive else text.lower()
            for keyword in rule.match.keywords:
                needle = keyword if rule.match.case_sensitive else keyword.lower()
                if needle in haystack:
                    found.append(keyword)
            return found
        if match_type == "regex" and rule.match.pattern:
            return [match.group(0) for match in re.finditer(rule.match.pattern, text, flags=self._re_flags(rule.match))]
        if match_type == "topic":
            topic_score = float(context.get("topic_score", 0.0))
            threshold = rule.match.min_score if rule.match.min_score is not None else 0.10
            return ["topic_score={0:.3f}".format(topic_score)] if topic_score < threshold else []
        return []

    def _re_flags(self, match_config: MatchConfig) -> int:
        return 0 if match_config.case_sensitive else re.IGNORECASE
