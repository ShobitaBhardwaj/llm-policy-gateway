from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class SchemaViolation:
    path: str
    message: str


class SimpleJsonSchemaValidator:
    def parse_and_validate(self, text: str, schema: Optional[Dict[str, Any]]) -> (Optional[Any], List[SchemaViolation]):
        try:
            instance = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, [SchemaViolation(path="$", message="Invalid JSON: {0}".format(exc.msg))]

        if schema is None:
            return instance, []
        return instance, self.validate_instance(instance, schema)

    def validate_instance(self, instance: Any, schema: Dict[str, Any], path: str = "$") -> List[SchemaViolation]:
        violations: List[SchemaViolation] = []

        allowed_types = schema.get("type")
        if allowed_types is not None and not self._matches_type(instance, allowed_types):
            return [
                SchemaViolation(
                    path=path,
                    message="Expected type {0}.".format(self._display_type(allowed_types)),
                )
            ]

        if "enum" in schema and instance not in schema["enum"]:
            violations.append(SchemaViolation(path=path, message="Value is not in enum."))

        if "const" in schema and instance != schema["const"]:
            violations.append(SchemaViolation(path=path, message="Value does not match const."))

        if isinstance(instance, dict):
            violations.extend(self._validate_object(instance, schema, path))
        elif isinstance(instance, list):
            violations.extend(self._validate_array(instance, schema, path))
        elif isinstance(instance, str):
            violations.extend(self._validate_string(instance, schema, path))
        elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
            violations.extend(self._validate_number(instance, schema, path))

        return violations

    def _validate_object(self, instance: Dict[str, Any], schema: Dict[str, Any], path: str) -> List[SchemaViolation]:
        violations: List[SchemaViolation] = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)

        for field_name in required:
            if field_name not in instance:
                violations.append(SchemaViolation(path=path, message="Missing required field '{0}'.".format(field_name)))

        for key, value in instance.items():
            next_path = "{0}.{1}".format(path, key)
            if key in properties:
                violations.extend(self.validate_instance(value, properties[key], next_path))
            elif additional is False:
                violations.append(SchemaViolation(path=next_path, message="Additional properties are not allowed."))
            elif isinstance(additional, dict):
                violations.extend(self.validate_instance(value, additional, next_path))

        return violations

    def _validate_array(self, instance: List[Any], schema: Dict[str, Any], path: str) -> List[SchemaViolation]:
        violations: List[SchemaViolation] = []
        items_schema = schema.get("items")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")

        if min_items is not None and len(instance) < min_items:
            violations.append(SchemaViolation(path=path, message="Expected at least {0} items.".format(min_items)))
        if max_items is not None and len(instance) > max_items:
            violations.append(SchemaViolation(path=path, message="Expected at most {0} items.".format(max_items)))

        if isinstance(items_schema, dict):
            for index, item in enumerate(instance):
                violations.extend(self.validate_instance(item, items_schema, "{0}[{1}]".format(path, index)))

        return violations

    def _validate_string(self, instance: str, schema: Dict[str, Any], path: str) -> List[SchemaViolation]:
        violations: List[SchemaViolation] = []
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")

        if min_length is not None and len(instance) < min_length:
            violations.append(SchemaViolation(path=path, message="String is shorter than minLength {0}.".format(min_length)))
        if max_length is not None and len(instance) > max_length:
            violations.append(SchemaViolation(path=path, message="String is longer than maxLength {0}.".format(max_length)))
        if pattern is not None and not re.search(pattern, instance):
            violations.append(SchemaViolation(path=path, message="String does not match pattern."))

        return violations

    def _validate_number(self, instance: float, schema: Dict[str, Any], path: str) -> List[SchemaViolation]:
        violations: List[SchemaViolation] = []
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        if minimum is not None and instance < minimum:
            violations.append(SchemaViolation(path=path, message="Number is lower than minimum {0}.".format(minimum)))
        if maximum is not None and instance > maximum:
            violations.append(SchemaViolation(path=path, message="Number is greater than maximum {0}.".format(maximum)))

        return violations

    def _matches_type(self, instance: Any, allowed_types: Any) -> bool:
        if isinstance(allowed_types, str):
            allowed = [allowed_types]
        elif isinstance(allowed_types, Sequence):
            allowed = list(allowed_types)
        else:
            return True

        type_checks = {
            "object": lambda value: isinstance(value, dict),
            "array": lambda value: isinstance(value, list),
            "string": lambda value: isinstance(value, str),
            "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
            "boolean": lambda value: isinstance(value, bool),
            "null": lambda value: value is None,
        }
        return any(type_checks.get(type_name, lambda _: True)(instance) for type_name in allowed)

    def _display_type(self, allowed_types: Any) -> str:
        if isinstance(allowed_types, str):
            return allowed_types
        if isinstance(allowed_types, Sequence):
            return ", ".join(str(item) for item in allowed_types)
        return "unknown"
