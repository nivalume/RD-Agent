"""Crypto factor R&D loop: from a hypothesis to the factors it is turned into.

The crypto scenario is proposal-only for now. :class:`CryptoFactorRDLoop` wires the scenario (the
market, its calendar and the data schemas), the hypothesis generator and the hypothesis-to-experiment
converter, so one loop is:

1. ``propose`` -- generate a hypothesis from the scenario and the history,
2. ``exp_gen`` -- turn that hypothesis into :class:`~rdagent.components.coder.factor_coder.factor.FactorTask` s,
3. ``record`` -- put the experiment in the trace so the next round does not propose its factors again.

There is no coder, runner or summarizer yet (see
:class:`rdagent.app.crypto_rd_loop.conf.CryptoPropSetting`), so the loop ends after ``exp_gen`` and the
factors are recorded as unevaluated rather than as a success.

Run it with::

    python -m rdagent.app.crypto_rd_loop.factor --loop_n 1

A session can be resumed from a dumped loop, the way ``qlib_rd_loop`` does it::

    python -m rdagent.app.crypto_rd_loop.factor --path <log>/__session__/1/0_propose --step_n 1
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

import fire
from dotenv import load_dotenv

# The CRYPTO_* settings usually live in the repository's .env, and conf.py builds
# CRYPTO_PROP_SETTING at import time; load the file before that import.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from rdagent.app.crypto_rd_loop.conf import CRYPTO_PROP_SETTING, CryptoPropSetting  # noqa: E402
from rdagent.core.conf import RD_AGENT_SETTINGS  # noqa: E402
from rdagent.core.proposal import ExperimentPlan, HypothesisFeedback, Trace  # noqa: E402
from rdagent.core.scenario import Scenario  # noqa: E402
from rdagent.core.utils import import_class  # noqa: E402
from rdagent.log import rdagent_logger as logger  # noqa: E402
from rdagent.utils.workflow import LoopBase, LoopMeta  # noqa: E402

NOT_EVALUATED = (
    "The factors were not evaluated: this loop wires the proposal side only (no coder, runner or "
    "summarizer), so they are kept as they are and not re-proposed in the next round."
)
"""The feedback recorded when no evaluation ran, so the factors stay usable for the next round."""


class CryptoFactorRDLoop(LoopBase, metaclass=LoopMeta):
    """Generate a hypothesis, turn it into factor tasks, record them; no coding and no evaluation."""

    def __init__(self, prop_setting: CryptoPropSetting | None = None) -> None:
        super().__init__()
        self.prop_setting = prop_setting if prop_setting is not None else CRYPTO_PROP_SETTING
        logger.log_object(self.prop_setting.model_dump(), tag="RDLOOP_SETTINGS")
        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")

        scen_cls = import_class(self.prop_setting.scen)
        self.scen: Scenario = scen_cls(with_runtime_environment=self.prop_setting.with_runtime_environment)
        logger.log_object(self.scen, tag="scenario")

        self.hypothesis_gen = import_class(self.prop_setting.hypothesis_gen)(self.scen)
        self.hypothesis2experiment = import_class(self.prop_setting.hypothesis2experiment)()
        self.plan: ExperimentPlan = {"features": {}, "feature_codes": {}}
        self.trace: Trace = Trace(scen=self.scen)

    def propose(self, prev_out: dict[str, Any]):
        """Generate the next hypothesis from the scenario and everything recorded so far."""
        hypothesis = self.hypothesis_gen.gen(self.trace, self.plan)
        logger.log_object(hypothesis, tag="hypothesis generation")
        return hypothesis

    def exp_gen(self, prev_out: dict[str, Any]):
        """Turn the hypothesis into the factors that implement it."""
        experiment = self.hypothesis2experiment.convert(prev_out["propose"], self.trace)
        logger.log_object(experiment.sub_tasks, tag="experiment generation")
        return experiment

    def record(self, prev_out: dict[str, Any]):
        """Put the experiment in the trace, unevaluated, so the next round can tell it apart."""
        experiment = prev_out["exp_gen"]
        feedback = HypothesisFeedback(reason=NOT_EVALUATED, decision=True, acceptable=True)
        self.trace.sync_dag_parent_and_hist((experiment, feedback), prev_out[self.LOOP_IDX_KEY])
        logger.log_object(feedback, tag="feedback")
        return feedback


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: bool = True,
    checkout_path: Optional[str] = None,
) -> None:
    """Run the crypto factor loop, or resume a dumped session.

    Parameters
    ----------
    path : str | None
        A dumped loop to resume; a fresh loop when omitted.
    step_n : int | None
        How many steps to run; all remaining ones when omitted.
    loop_n : int | None
        How many loops to run; forever when omitted.
    all_duration : str | None
        Maximum duration, in the format the timer accepts.
    checkout : bool
        Reset the session's working tree before resuming.
    checkout_path : str | None
        Check out to this path instead of the dumped one.
    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    loop = (
        CryptoFactorRDLoop(CRYPTO_PROP_SETTING)
        if path is None
        else CryptoFactorRDLoop.load(path, checkout=checkout)
    )
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
