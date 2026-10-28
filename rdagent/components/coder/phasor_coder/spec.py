"""The artefact a phasor factor is: one JSON object, expression mode or python mode.

The prompt asks the model for exactly this shape. This module checks that shape
and the coder's own limits (how large a parameter sweep may be, which panel
columns may be read). It does not parse the expression language: that is
``phasor.Expression.compile`` for a plain expression and
``phasor.FactorGrid.compile`` when the expression has ``{name}`` placeholders.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_PARAMS = 2
DEFAULT_MAX_CASES = 6

_FENCE = chr(96) * 3
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DEFINE = re.compile(r"^def\s+calculate\s*\(", re.MULTILINE)

EXPRESSION_KEYS = frozenset({"mode", "expression", "params", "fields", "reason"})
PYTHON_KEYS = frozenset({"mode", "code", "fields", "reason"})


class FactorSpecError(ValueError):
    """The model's answer cannot be used as a factor."""


@dataclass(frozen=True)
class FactorSpec:
    """One factor implementation, already validated."""

    mode: str
    fields: tuple[str, ...]
    reason: str = ""
    expression: str | None = None
    params: dict[str, list[float]] = field(default_factory=dict)
    code: str | None = None

    @property
    def cases(self) -> int:
        """Number of grid cases the evaluator will run."""
        count = 1
        for values in self.params.values():
            count *= len(values)
        return count

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"mode": self.mode, "fields": list(self.fields), "reason": self.reason}
        if self.mode == "expression":
            data["expression"] = self.expression
            if self.params:
                data["params"] = self.params
        else:
            data["code"] = self.code
        return data

    def to_files(self) -> dict[str, str]:
        """The file to write into a workspace. The spec is the whole artefact."""
        return {
            "factor.json": json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
        }


def parse_factor_spec(
    text: str,
    *,
    available_fields: list[str] | tuple[str, ...] | None = None,
    max_params: int = DEFAULT_MAX_PARAMS,
    max_cases: int = DEFAULT_MAX_CASES,
) -> FactorSpec:
    """Parse and validate one model response. Raises FactorSpecError."""
    data = _extract_json(text)
    return validate_factor_spec(
        data,
        available_fields=available_fields,
        max_params=max_params,
        max_cases=max_cases,
    )


def validate_factor_spec(
    data: dict[str, Any],
    *,
    available_fields: list[str] | tuple[str, ...] | None = None,
    max_params: int = DEFAULT_MAX_PARAMS,
    max_cases: int = DEFAULT_MAX_CASES,
) -> FactorSpec:
    """Validate an already parsed mapping and turn it into a FactorSpec."""
    errors: list[str] = []
    mode = data.get("mode")
    if mode not in ("expression", "python"):
        raise FactorSpecError('"mode" must be "expression" or "python"')

    allowed = EXPRESSION_KEYS if mode == "expression" else PYTHON_KEYS
    unknown = sorted(set(data) - allowed)
    if unknown:
        errors.append(f"unknown keys: {', '.join(unknown)}")

    fields = _validate_fields(data.get("fields"), available_fields, errors)
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        errors.append('"reason" must be a string')

    if mode == "expression":
        params = _normalise_params(data.get("params"), errors)
        expression = _validate_expression(
            data.get("expression"),
            params,
            available_fields,
            errors,
            max_params=max_params,
            max_cases=max_cases,
        )
        if errors:
            raise FactorSpecError("\n".join(f"- {error}" for error in errors))
        return FactorSpec(
            mode=mode, fields=fields, reason=reason, expression=expression, params=params
        )

    code = _validate_code(data.get("code"), errors)
    if errors:
        raise FactorSpecError("\n".join(f"- {error}" for error in errors))
    return FactorSpec(mode=mode, fields=fields, reason=reason, code=code)


def _extract_json(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    if _FENCE in text:
        after = text.split(_FENCE, 1)[1]
        if "json" in after[:8].split("\n", 1)[0]:
            after = after.split("\n", 1)[1]
        candidates.append(after.split(_FENCE, 1)[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    preview = text.strip().replace("\n", " ")[:200]
    raise FactorSpecError(f"the response is not a JSON object: {preview!r}")


def _validate_fields(
    raw: Any, available: list[str] | tuple[str, ...] | None, errors: list[str]
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        errors.append('"fields" must be a non-empty list of field names')
        return ()
    names: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name:
            errors.append('"fields" may only contain non-empty strings')
            continue
        if name.startswith("$"):
            errors.append(f'field "{name}" must be written without the leading dollar sign')
            continue
        names.append(name)
    if available:
        missing = [name for name in names if name not in available]
        if missing:
            errors.append(
                f"field(s) not available: {', '.join(missing)}; available: {', '.join(available)}"
            )
    return tuple(dict.fromkeys(names))


def _validate_expression(
    raw: Any,
    params: dict[str, list[float]],
    available: list[str] | tuple[str, ...] | None,
    errors: list[str],
    *,
    max_params: int,
    max_cases: int,
) -> str:
    if not isinstance(raw, str) or not raw.strip():
        errors.append('"expression" must be a non-empty string')
        return ""
    expression = raw.strip()

    # The sweep size is the coder's budget. The expression language itself,
    # including `{name}` substitution, is checked by phasor below.
    placeholders = _PLACEHOLDER.findall(expression)
    aligned = True
    for name in placeholders:
        if name not in params:
            errors.append(f'placeholder {{{name}}} has no case in "params"')
            aligned = False
    for name in params:
        if name not in placeholders:
            errors.append(f'parameter "{name}" is not used as a placeholder in "expression"')
            aligned = False
    if len(params) > max_params:
        errors.append(f"at most {max_params} parameters are allowed, got {len(params)}")
        aligned = False
    count = 1
    for values in params.values():
        count *= len(values)
    if count > max_cases:
        errors.append(f"at most {max_cases} cases are allowed, got {count}")
        aligned = False
    if aligned:
        _check_with_phasor(expression, params, available, errors)
    return expression


def _check_with_phasor(
    expression: str,
    params: dict[str, list[float]],
    available: list[str] | tuple[str, ...] | None,
    errors: list[str],
) -> None:
    """Compile the expression with phasor and apply the panel's field list.

    ``Expression.compile`` parses one expression. ``FactorGrid.compile``
    substitutes every ``{name}`` and compiles each case, which is what rejects
    an unclosed placeholder, a bad window, or an unknown operator.
    """
    try:
        import phasor
    except ImportError:
        errors.append("phasor is not installed; the expression cannot be checked")
        return
    try:
        if params:
            compiled = phasor.FactorGrid.compile({"factor": expression}, **params)
        else:
            compiled = phasor.Expression.compile(expression)
    except Exception as error:  # noqa: BLE001 - the engine reports its own error type
        errors.append(f"phasor rejected the expression: {error}")
        return
    if compiled.uses_future:
        errors.append("the expression reads the future: uses_future is True")
    if not available:
        return
    allowed = {name.lower() for name in available}
    unknown = [name for name in compiled.required_fields if name.lower() not in allowed]
    if unknown:
        errors.append(
            "field(s) not available: "
            + ", ".join("$" + name for name in unknown)
            + "; available: "
            + ", ".join(available)
        )


def _normalise_params(raw: Any, errors: list[str]) -> dict[str, list[float]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append('"params" must be an object of name -> list of numbers')
        return {}
    params: dict[str, list[float]] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            errors.append(f'invalid parameter name "{name}"')
            continue
        if not isinstance(values, (list, tuple)) or not values:
            errors.append(f'parameter "{name}" must be a non-empty list of numbers')
            continue
        numbers: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f'parameter "{name}" may only contain numbers, got {value!r}')
                continue
            if not math.isfinite(value) or value < 0:
                errors.append(f'parameter "{name}" must be finite and non-negative, got {value!r}')
                continue
            numbers.append(value)
        if numbers:
            params[name] = numbers
    return params


def _validate_code(raw: Any, errors: list[str]) -> str:
    if not isinstance(raw, str) or not raw.strip():
        errors.append('"code" must be a non-empty string')
        return ""
    code = raw
    if not _DEFINE.search(code):
        errors.append('the code must define "def calculate(fields, index=None, symbols=None)"')
    try:
        compile(code, "factor.py", "exec")
    except SyntaxError as error:
        errors.append(f"the code does not parse: {error.msg} (line {error.lineno})")
    return code
