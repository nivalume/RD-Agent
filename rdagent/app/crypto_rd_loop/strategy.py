"""Crypto strategy mining: propose a mechanism, code it as positions, backtest it, learn from it.

One round:

1. ``direct_exp_gen`` -- a hypothesis and the strategies that test it (the strategy proposal);
2. ``coding`` -- each strategy as a phasor expression or numpy code, checked on the debug panel
   (bounds, not flat, no look-ahead);
3. ``running`` -- the backtest on the full panel: volatility targeting, costs, funding, train /
   select / lockbox, robustness and diagnostics;
4. ``feedback`` -- the hard gates and the best-so-far comparison decide; the model explains;
5. ``record``.

The lockbox is never shown to the model (``CRYPTO_STRATEGY_HIDE_TEST``); read it with the
report::

    jarvis-data panel --interval 4h --top 100 --out ~/jarvis-data/strategy/4h   # the data, once
    python -m rdagent.app.crypto_rd_loop.strategy --loop_n 20
    python -m rdagent.scenarios.crypto.strategy.report log/<session>

Resume a dumped session::

    python -m rdagent.app.crypto_rd_loop.strategy --path log/<session>/__session__/3/0_direct_exp_gen --step_n 1
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

import fire
from dotenv import load_dotenv

# The settings below are built when their modules are imported; load the repository's .env first.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from pydantic_settings import SettingsConfigDict  # noqa: E402

from rdagent.components.workflow.conf import BasePropSetting  # noqa: E402
from rdagent.components.workflow.rd_loop import RDLoop  # noqa: E402
from rdagent.log import rdagent_logger as logger  # noqa: E402

STRATEGY = "rdagent.scenarios.crypto.strategy"


class CryptoStrategyPropSetting(BasePropSetting):
    """The classes the strategy loop wires; the evaluation itself is in ``CRYPTO_STRATEGY_`` settings."""

    model_config = SettingsConfigDict(env_prefix="CRYPTO_STRATEGY_LOOP_", protected_namespaces=())

    scen: str = f"{STRATEGY}.scenario.CryptoStrategyBacktestScenario"
    hypothesis_gen: str = "rdagent.scenarios.crypto.proposal.strategy_proposal.CryptoStrategyHypothesisGen"
    hypothesis2experiment: str = (
        "rdagent.scenarios.crypto.proposal.strategy_proposal.CryptoStrategyHypothesis2Experiment"
    )
    coder: str = "rdagent.components.coder.strategy_coder.StrategyCoSTEER"
    runner: str = f"{STRATEGY}.developer.StrategyRunner"
    summarizer: str = f"{STRATEGY}.developer.StrategyExperiment2Feedback"


class CryptoStrategyRDLoop(RDLoop):
    def running(self, prev_out: dict[str, Any]):
        exp = prev_out["coding"]
        # Every earlier strategy is a trial, accepted or not: the deflated Sharpe counts them all.
        exp.prior_trial_sharpes = [
            value for earlier, _ in self.trace.hist for value in (getattr(earlier, "trial_sharpes", None) or [])
        ]
        exp = self.runner.develop(exp)
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: Optional[str] = None,
    checkout: bool = True,
) -> None:
    """Run the strategy loop, or resume it from ``path``."""
    loop = (
        CryptoStrategyRDLoop(CryptoStrategyPropSetting())
        if path is None
        else CryptoStrategyRDLoop.load(path, checkout=checkout)
    )
    asyncio.run(loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
