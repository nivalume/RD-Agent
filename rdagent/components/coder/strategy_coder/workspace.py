from __future__ import annotations

import json
from typing import Any

from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.crypto.strategy.run import SPEC_FILE


class StrategyWorkspace(FBWorkspace):
    """A workspace holding one strategy as ``strategy.json``: the position expression, or the
    numpy module in its ``code`` field. The prompts quote that file, not ``.py`` files."""

    def hash_func(self, data_type: str = "Debug") -> str | None:
        return md5_hash(data_type + (self.file_dict.get(SPEC_FILE) or "")) or None

    @property
    def all_codes(self) -> str:
        spec = self.file_dict.get(SPEC_FILE)
        return self._format_code_dict({SPEC_FILE: spec}) if spec else ""

    @property
    def spec(self) -> dict[str, Any] | None:
        """The parsed ``strategy.json``, or None when it is missing or malformed."""
        raw = self.file_dict.get(SPEC_FILE)
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
