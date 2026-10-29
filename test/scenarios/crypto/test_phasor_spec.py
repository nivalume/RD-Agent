"""Expression checks go through phasor, not a second parser."""

import json

import pytest

from rdagent.components.coder.phasor_coder.spec import FactorSpecError, parse_factor_spec

pytestmark = pytest.mark.offline

pytest.importorskip("phasor")


def _expression(expression: str, params: dict | None = None, fields: list[str] | None = None) -> dict:
    data = {
        "mode": "expression",
        "expression": expression,
        "fields": fields or ["close"],
        "reason": "test",
    }
    if params is not None:
        data["params"] = params
    return data


def test_plain_expression_compiles() -> None:
    spec = parse_factor_spec(
        json.dumps(_expression("Mean($close, 20) / $close - 1")),
        available_fields=["close", "volume"],
    )
    assert spec.mode == "expression"
    assert spec.fields == ("close",)


def test_parameter_grid_compiles_every_case() -> None:
    spec = parse_factor_spec(
        json.dumps(_expression("Mean($close, {n}) / $close - 1", {"n": [5, 20]})),
        available_fields=["close"],
    )
    assert spec.cases == 2
    assert spec.params["n"] == [5, 20]


def test_unknown_operator_is_phasors_error() -> None:
    with pytest.raises(FactorSpecError, match="phasor rejected the expression"):
        parse_factor_spec(json.dumps(_expression("NotAnOperator($close)")), available_fields=["close"])


def test_fractional_window_is_phasors_error() -> None:
    with pytest.raises(FactorSpecError, match="phasor rejected the expression"):
        parse_factor_spec(json.dumps(_expression("Mean($close, 1.5)")), available_fields=["close"])


def test_future_read_is_rejected() -> None:
    with pytest.raises(FactorSpecError, match="uses_future"):
        parse_factor_spec(json.dumps(_expression("Ref($close, -1)")), available_fields=["close"])


def test_files_are_the_spec_alone() -> None:
    spec = parse_factor_spec(
        json.dumps(_expression("Mean($close, 20) / $close - 1")),
        available_fields=["close"],
    )
    files = spec.to_files()
    assert list(files) == ["factor.json"]
    stored = json.loads(files["factor.json"])
    assert stored["expression"] == "Mean($close, 20) / $close - 1"
    assert "params" not in stored


def test_python_code_stays_inside_factor_json() -> None:
    code = "def calculate(fields, index=None, symbols=None):\n    return fields['close']\n"
    spec = parse_factor_spec(
        json.dumps({"mode": "python", "code": code, "fields": ["close"], "reason": "test"}),
        available_fields=["close"],
    )
    files = spec.to_files()
    assert list(files) == ["factor.json"]
    assert json.loads(files["factor.json"])["code"] == code


def test_required_fields_come_from_the_compiled_expression() -> None:
    with pytest.raises(FactorSpecError, match=r"\$funding_rate"):
        parse_factor_spec(
            json.dumps(_expression("Mean($funding_rate, 5)", fields=["funding_rate"])),
            available_fields=["close", "volume"],
        )
