from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_guard.models import GuardrailViolation


class GuardrailError(Exception):
    def __init__(
        self,
        http_status: int,
        error_code: str,
        message: str,
        violations: Optional[List[GuardrailViolation]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.error_code = error_code
        self.message = message
        self.violations = violations or []
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "violations": [violation.model_dump() for violation in self.violations],
                "details": self.details,
            }
        }


class PolicyError(Exception):
    """Raised when policy configuration is missing or invalid."""
