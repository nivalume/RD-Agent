# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                                |
| -------------- | ---------------------------------------------------------------------------|
| "daily_pv.h5"  | Daily bars of crypto perpetuals: prices, volumes, trade counts, funding.    |

The index is a ("datetime", "instrument") MultiIndex. `datetime` is a UTC day: crypto trades every
day, weekends included, and the value of a day is known at its end (00:00 UTC of the next day).
`instrument` is a Binance USD-M perpetual quoted in USDT, named by its symbol (`BTCUSDT`). The
instruments are every one that was ever in the trading pool, delisted ones included, over their
whole history: an instrument may have rows before it entered the pool and stops having rows when it
was delisted. A value the exchange did not publish is NaN.

The table holds prices, volumes and the funding rate. Open interest and long/short ratios are not
part of it.

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

## Funding rate (perpetuals only)
$funding_rate: the funding settled on that day, summed over its settlements (three for an 8-hour
funding interval, six for a 4-hour one), as a fraction of the position's value: 0.0003 is 0.03% of
the position paid by longs to shorts (0.01% per settlement). Negative means shorts paid longs. NaN
on a day the exchange settled nothing. The column is named `$funding_rate`, dollar sign included.

## Other
$factor: always 1: perpetual prices need no adjustment.
