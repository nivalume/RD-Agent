"""Cross-sectional factor and model research on crypto perpetuals, run on qlib.

The loop is RD-Agent's qlib loop -- its factor and model coders, its runners and its feedback --
pointed at a crypto data directory instead of the CN one:

* the data is a qlib directory written by ``jarvis-history qlib`` (``~/.qlib/qlib_data/crypto_data``
  by default): every USD-M perpetual quoted in USDT, delisted ones included, a row a UTC day, and
  monthly ``top<N>`` pools chosen from the 30 days before each month;
* the scenarios are this package's own classes, not the qlib ones: they reuse the qlib prompts that
  do not depend on the market (code interfaces, model output, data folder description), describe the
  crypto market with the crypto prompts of :mod:`rdagent.scenarios.crypto.experiment`, and the qlib
  setting (pool, label, costs, backtest) with ``prompts.yaml`` here;
* the workflow configs under ``factor_template`` and ``model_template`` are the qlib ones with a 24/7
  calendar, no price limits, fractional sizes, perpetual fees and annualisation over 365 days;
* :class:`~rdagent.scenarios.crypto.qlib.workspace.CryptoQlibFBWorkspace` runs them in docker or conda,
  as ``MODEL_CoSTEER_ENV_TYPE`` says, with the settings of :mod:`rdagent.scenarios.crypto.qlib.conf`.

Start a loop with ``python -m rdagent.app.crypto_rd_loop.qlib factor`` (or ``model``, ``quant``).
"""
