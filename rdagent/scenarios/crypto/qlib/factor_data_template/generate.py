"""Write the price-volume tables the factor coder reads, from the crypto qlib data directory.

Run inside the qlib environment (docker or conda) with these variables, all optional:
    crypto_provider_uri, crypto_market, debug_instruments, debug_start

The work is in ``main()``, run only under ``if __name__ == "__main__"``: qlib reads the data with a
joblib process pool, and on macOS (and Windows) the pool starts its workers with spawn, which
re-imports this file as ``__mp_main__``. Without the guard every worker runs the whole script again,
spawning its own pool, and the run never ends.
"""

import os

import qlib

FIELDS = [
    "$open", "$close", "$high", "$low", "$volume", "$amount", "$vwap", "$count",
    "$taker_buy_volume", "$taker_buy_amount", "$funding_rate", "$factor",
]  # fmt: skip


def main() -> None:
    qlib.init(provider_uri=os.environ.get("crypto_provider_uri", "~/.qlib/qlib_data/crypto_data"))

    from qlib.data import D  # noqa: PLC0415

    # Every instrument that was ever in the pool, over its whole history: a factor may look back
    # before the day an instrument joins the pool.
    market = os.environ.get("crypto_market", "top100")
    pool = D.instruments(market)
    members = sorted(D.list_instruments(pool, as_list=True))
    print(f"Reading {len(FIELDS)} fields of {len(members)} {market} instruments", flush=True)
    data = D.features(D.instruments("all"), FIELDS, freq="day")
    data = data.loc[data.index.get_level_values("instrument").isin(members)]
    data = data.swaplevel().sort_index()
    data.to_hdf("./daily_pv_all.h5", key="data")
    print(f"Wrote daily_pv_all.h5: {data.shape[0]} rows", flush=True)

    # A small table the coder runs a factor on first: the most traded instruments of the pool.
    debug_start = os.environ.get("debug_start", "2024-01-01")
    size = int(os.environ.get("debug_instruments", "30"))
    recent = data.loc[debug_start:]
    traded = recent["$amount"].groupby(level="instrument").mean().sort_values(ascending=False)
    keep = traded.index[:size]
    debug = recent.loc[recent.index.get_level_values("instrument").isin(keep)]
    debug.to_hdf("./daily_pv_debug.h5", key="data")
    print(f"Wrote daily_pv_debug.h5: {debug.shape[0]} rows, {len(keep)} instruments", flush=True)


if __name__ == "__main__":
    main()
