# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                                  |
| -------------- | -----------------------------------------------------------------------------|
| "daily_pv.h5"  | Daily bars, funding, open interest and positioning of crypto perpetuals.     |

The index is a ("datetime", "instrument") MultiIndex. `datetime` is a UTC day: crypto trades every
day, weekends included, and the value of a day is known at its end (00:00 UTC of the next day).
`instrument` is a Binance USD-M perpetual quoted in USDT, named by its symbol (`BTCUSDT`). The
instruments are every one that was ever in the trading pool, delisted ones included, over their
whole history: an instrument may have rows before it entered the pool and stops having rows when it
was delisted. A value the exchange did not publish is NaN.

# For different data, We have some basic knowledge for them

## Daily bars (1d klines, UTC day)
$open: price of the first trade of the day.
$close: price of the last trade of the day.
$high: highest price of the day.
$low: lowest price of the day.
$volume: traded quantity of the day, in the base asset (contracts of 1 unit of the coin).
$amount: traded value of the day, in USDT.
$vwap: volume-weighted average price of the day ($amount / $volume).
$count: number of trades of the day.
$taker_buy_volume: the part of $volume bought by takers (aggressive buyers).
$taker_buy_amount: the part of $amount bought by takers.

## Funding
$funding_rate: the funding rates settled during the day, summed (usually three settlements, at
00:00, 08:00 and 16:00 UTC). Positive: longs paid shorts. A fraction, not a percentage.

## Open interest (from 5-minute snapshots; NaN where not downloaded)
$open_interest: open interest at the last snapshot of the day, in the base asset.
$open_interest_value: open interest at the last snapshot of the day, in USDT.

## Positioning (from 5-minute snapshots; NaN where not downloaded)
$top_trader_account_ls: long/short ratio of the top traders' accounts.
$top_trader_position_ls: long/short ratio of the top traders' positions.
$global_account_ls: long/short ratio of all accounts.
$taker_buy_sell_ratio: taker buy volume over taker sell volume.

## Other
$factor: always 1: perpetual prices need no adjustment.
