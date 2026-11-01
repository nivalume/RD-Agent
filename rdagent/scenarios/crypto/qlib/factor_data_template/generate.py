"""Write the price-volume tables the factor coder reads, from the crypto qlib data directory.

Run inside the qlib environment (docker or conda) with these variables, all optional:
    crypto_provider_uri, crypto_market, debug_instruments, debug_start, debug_end
"""

import os

import qlib

qlib.init(provider_uri=os.environ.get("crypto_provider_uri", "~/.qlib/qlib_data/crypto_data"))

from qlib.data import D  # noqa: E402

FIELDS = [
    "$open", "$close", "$high", "$low", "$volume", "$amount", "$vwap", "$count",
    "$taker_buy_volume", "$taker_buy_amount", "$funding_rate",
    "$open_interest", "$open_interest_value",
    "$top_trader_account_ls", "$top_trader_position_ls", "$global_account_ls", "$taker_buy_sell_ratio",
    "$factor",
]  # fmt: skip

# Every instrument that was ever in the pool, over its whole history: a factor may look back
# before the day an instrument joins the pool.
market = os.environ.get("crypto_market", "top100")
pool = D.instruments(market)
members = sorted(D.list_instruments(pool, as_list=True))
data = D.features(D.instruments("all"), FIELDS, freq="day")
data = data.loc[data.index.get_level_values("instrument").isin(members)]
data = data.swaplevel().sort_index()
data.to_hdf("./daily_pv_all.h5", key="data")

# A small table the coder runs a factor on first: the most traded instruments of the pool, over a
# year that ends with the fitting period, so nothing the coder reports comes from later data.
debug_start = os.environ.get("debug_start", "2022-01-01")
debug_end = os.environ.get("debug_end", "2022-12-31")
size = int(os.environ.get("debug_instruments", "30"))
recent = data.loc[debug_start:debug_end]
traded = recent["$amount"].groupby(level="instrument").mean().sort_values(ascending=False)
keep = traded.index[:size]
debug = recent.loc[recent.index.get_level_values("instrument").isin(keep)]
debug.to_hdf("./daily_pv_debug.h5", key="data")
