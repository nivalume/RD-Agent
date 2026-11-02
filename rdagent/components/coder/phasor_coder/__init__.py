"""The phasor factor coder: it writes one factor, expression or numpy, per task.

The public entry point is :class:`PhasorCoSTEER`, which plugs the phasor prompts,
the spec validator and the code-generation evaluator into the CoSTEER loop::

    coder = PhasorCoSTEER(scen=CryptoFactorScenario())
    exp = coder.develop(exp)

A task is a :class:`PhasorFactorTask`; its workspace holds ``factor.json``.

Render the implementation prompt and call the model once, for debugging::

    python -m rdagent.components.coder.phasor_coder
    python -m rdagent.components.coder.phasor_coder --print-only
"""

# The names load on first use: ``spec`` is imported by the strategy runner's child process,
# which must not pay for the CoSTEER loop and the LLM backend.
_EXPORTS = {
    "PhasorCoSTEER": "coder",
    "PhasorCoSTEERSettings": "config",
    "PHASOR_COSTEER_SETTINGS": "config",
    "PhasorFactorTask": "task",
    "PhasorFactorWorkspace": "workspace",
    "PhasorFactorEvaluatorForCoder": "evaluators",
    "PhasorMultiProcessEvolvingStrategy": "evolving_strategy",
    "FactorSpec": "spec",
    "FactorSpecError": "spec",
    "parse_factor_spec": "spec",
    "validate_factor_spec": "spec",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)


__all__ = [
    "PhasorCoSTEER",
    "PhasorCoSTEERSettings",
    "PHASOR_COSTEER_SETTINGS",
    "PhasorFactorTask",
    "PhasorFactorWorkspace",
    "PhasorFactorEvaluatorForCoder",
    "PhasorMultiProcessEvolvingStrategy",
    "FactorSpec",
    "FactorSpecError",
    "parse_factor_spec",
    "validate_factor_spec",
]
