from __future__ import annotations

import json
from typing import Any

from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_utils import md5_hash


class PhasorFactorWorkspace(FBWorkspace):
    """A workspace holding one factor as ``factor.json``.

    The spec is the whole artefact: an expression, or the numpy module in its
    ``code`` field. :meth:`FBWorkspace.all_codes` keeps ``.py`` files only, so the
    prompts would otherwise quote an empty implementation.
    """

    def hash_func(self, data_type: str = "Debug") -> str | None:
        return md5_hash(data_type + (self.file_dict.get("factor.json") or "")) or None

    @property
    def all_codes(self) -> str:
        spec = self.file_dict.get("factor.json")
        if not spec:
            return ""
        return self._format_code_dict({"factor.json": spec})

    @property
    def spec(self) -> dict[str, Any] | None:
        """The parsed ``factor.json``, or None when it is missing or malformed."""
        raw = self.file_dict.get("factor.json")
        if not raw:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
