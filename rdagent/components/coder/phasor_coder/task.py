from __future__ import annotations

from rdagent.components.coder.factor_coder.factor import FactorTask


class PhasorFactorTask(FactorTask):
    """A factor to implement for phasor.

    A factor is evaluated on one bar frequency only, so ``frequency`` describes
    the grid the factor is computed on: a window of N in the expression means N
    bars of that grid. It stays empty when the scenario fixes the grid in its data
    instead, in which case the prompt is silent about it.
    """

    def __init__(
        self,
        *args,
        frequency: str | None = None,
        fields: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frequency = frequency
        self.fields = fields

    def get_task_information(self) -> str:
        text = super().get_task_information()
        if self.frequency:
            text += (
                f"\nfrequency: {self.frequency}"
                f" (one bar is one {self.frequency}; a window of N is N such bars)"
            )
        if self.fields:
            text += f"\nfields: {', '.join(self.fields)}"
        return text
