from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Detection:
    category: str
    start: int
    end: int
    value: str
    reason: str


class InjectionDetector:
    PATTERNS: Sequence[Tuple[str, str]] = (
        ("ignore_previous", r"ignore (all|any|the) (previous|prior) instructions"),
        ("reveal_system_prompt", r"(show|reveal|print).{0,40}(system prompt|developer message|hidden prompt)"),
        ("bypass_guardrails", r"(bypass|disable|override).{0,30}(guardrail|safety|policy|restriction)"),
        ("jailbreak", r"\bjailbreak\b"),
        ("role_override", r"(act as|pretend to be).{0,30}(system|developer|administrator|root)"),
        ("secret_exfiltration", r"(list|dump|leak|exfiltrate).{0,30}(secret|token|key|credential)"),
    )

    def detect(self, text: str) -> List[str]:
        lowered = text.lower()
        hits = []
        for label, pattern in self.PATTERNS:
            if re.search(pattern, lowered, re.IGNORECASE | re.DOTALL):
                hits.append(label)
        return hits


class PIIDetector:
    EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?(?:\(?\d{3}\)?[-. ]?){2}\d{4}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
    PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
    API_KEY_PATTERN = re.compile(
        r"\b(?:sk-[A-Za-z0-9]{16,}|api[_-]?key\s*[:=]\s*[A-Za-z0-9_\-]{8,}|bearer\s+[A-Za-z0-9_\-.]{12,})",
        re.IGNORECASE,
    )
    PASSWORD_PATTERN = re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE)

    def detect(self, text: str) -> List[Detection]:
        findings: List[Detection] = []
        findings.extend(self._match_pattern(text, self.EMAIL_PATTERN, "email", "Email address detected."))
        findings.extend(self._match_pattern(text, self.PHONE_PATTERN, "phone", "Phone number detected."))
        findings.extend(self._match_pattern(text, self.SSN_PATTERN, "government_id", "Government ID detected."))
        findings.extend(self._match_pattern(text, self.PRIVATE_KEY_PATTERN, "private_key", "Private key detected."))
        findings.extend(self._match_pattern(text, self.API_KEY_PATTERN, "credential", "API credential detected."))
        findings.extend(self._match_pattern(text, self.PASSWORD_PATTERN, "credential", "Password assignment detected."))
        findings.extend(self._detect_credit_cards(text))
        return findings

    def redact(self, text: str, detections: Optional[Iterable[Detection]] = None) -> str:
        hits = sorted(detections or self.detect(text), key=lambda item: item.start, reverse=True)
        redacted = text
        for hit in hits:
            replacement = "[REDACTED:{0}]".format(hit.category.upper())
            redacted = redacted[: hit.start] + replacement + redacted[hit.end :]
        return redacted

    def _match_pattern(self, text: str, pattern: re.Pattern, category: str, reason: str) -> List[Detection]:
        return [
            Detection(
                category=category,
                start=match.start(),
                end=match.end(),
                value=match.group(0),
                reason=reason,
            )
            for match in pattern.finditer(text)
        ]

    def _detect_credit_cards(self, text: str) -> List[Detection]:
        hits: List[Detection] = []
        for match in self.CREDIT_CARD_PATTERN.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 19 and self._passes_luhn(digits):
                hits.append(
                    Detection(
                        category="payment_card",
                        start=match.start(),
                        end=match.end(),
                        value=match.group(0),
                        reason="Payment card number detected.",
                    )
                )
        return hits

    def _passes_luhn(self, digits: str) -> bool:
        checksum = 0
        parity = len(digits) % 2
        for index, character in enumerate(digits):
            number = int(character)
            if index % 2 == parity:
                number *= 2
                if number > 9:
                    number -= 9
            checksum += number
        return checksum % 10 == 0


class ToxicityDetector:
    TOXIC_PATTERNS: Sequence[str] = (
        r"\bidiot\b",
        r"\bstupid\b",
        r"\bworthless\b",
        r"kill yourself",
        r"\bhate (you|them|him|her)\b",
    )

    def detect(self, text: str) -> List[str]:
        lowered = text.lower()
        return [pattern for pattern in self.TOXIC_PATTERNS if re.search(pattern, lowered, re.IGNORECASE)]


class TopicDetector:
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "we",
        "with",
        "you",
        "your",
    }

    def score(self, prompt_text: str, response_text: str) -> float:
        prompt_tokens = self._tokenize(prompt_text)
        response_tokens = self._tokenize(response_text)
        if not prompt_tokens:
            return 1.0
        overlap = prompt_tokens.intersection(response_tokens)
        return len(overlap) / float(len(prompt_tokens))

    def _tokenize(self, text: str) -> set:
        tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9_]{3,}", text)}
        return {token for token in tokens if token not in self.STOP_WORDS}


class CitationDetector:
    TEXT_PATTERNS: Sequence[str] = (
        r"https?://",
        r"\[[0-9]+\]",
        r"\bSources?:",
        r"\bReferences?:",
    )
    CITATION_KEYS = ("sources", "citations", "references")

    def has_citation(self, text: str, structured_output: Optional[Any] = None) -> bool:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.TEXT_PATTERNS):
            return True
        if isinstance(structured_output, dict):
            for key in self.CITATION_KEYS:
                value = structured_output.get(key)
                if isinstance(value, list) and value:
                    return True
                if isinstance(value, str) and value.strip():
                    return True
        return False
