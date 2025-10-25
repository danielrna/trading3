# file: etf_cross_sectional_momentum_v2.py
# Python 3.10+
# Deps: pandas, numpy, yfinance, matplotlib (optional)

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------- Config -------------------------------- #

SYMBOLS: List[str] = [
    "VOO",  # S&P 500
    "QQQ",  # Nasdaq 100
    "IWM",  # Russell 2000
    "DIA",  # Dow Jones
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLI",  # Industrials
    "XLU",  # Utilities
    "XLE",  # Energy
    "XLRE", # Real Estate
    "XLB",  # Materials
    "XLY",  # (duplicate guarded below)
]
SYMBOLS = list(dict.fromkeys(SYMBOLS))

BASE_SYMBOL = "VOO"              # used for signals/stats
YF_PERIOD = "max"

START_CAPITAL = 100_000.0
N_POSITIONS = 5
LOOKBACK_DAYS = 252              # 12-month momentum
SKIP_RECENT_DAYS = 21            # 1-month skip (12-1)
MIN_HISTORY_DAYS = LOOKBACK_DAYS + SKIP_RECENT_DAYS + 10

VOL_LOOKBACK = 20
MAX_WEIGHT_PER_ASSET = 0.25
MIN_WEIGHT_PER_ASSET = 0.0

# Costs
SLIPPAGE_BPS = 1                 # per side
FEE_BPS = 1                      # per side

# Liquidity guards
MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000

QTY_DECIMALS = 4

# Benchmark mode: "VOO" | "QQQ" | "EQUAL" | "TOP5_STATIC"
BENCHMARK_MODE = "EQUAL"

OUT_DIR = "./out"

# ------------------------------ Data Utils ------------------------------ #

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    if not symbols:
        raise ValueError("Empty symbol list")
    raw = yf.download(
        symbols, period=period, interval="1d",
        group_by="ticker", auto_adjust=False, threads=True, progress=False
    )
    data: Dict[str, pd.DataFrame] = {}

    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
        }).copy()
        out = out.dropna(subset=["open", "high", "low", "close", "volume"])
        bad = (
                (out["open"] <= 0) | (out["high"] <= 0) |
                (out["low"] <= 0)  | (out["close"] <= 0) | (out["volume"] < 0)
        )
        out = out[~bad]
        out.index.name = "date"
        return out[["open", "high", "low", "close", "volume"]]

    if isinstance(raw.columns, pd.MultiIndex):
        for s in symbols:
            if s not in raw.columns.get_level_values(0):
                raise ValueError(f"No data returned for {s}")
            data[s] = clean_df(raw[s].copy())
    else:
        s = symbols[0]
        data[s] = clean_df(raw)

    base = data.get(BASE_SYMBOL, None)
    if base is None:
        raise ValueError(f"BASE_SYMBOL {BASE_SYMBOL} must be in SYMBOLS and returned by Yahoo")
    base_idx = base.index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

    data = {s: df for s, df in data.items() if len(df) >= MIN_HISTORY_DAYS}
    if BASE_SYMBOL not in data:
        raise ValueError(f"BASE_SYMBOL {BASE_SYMBOL} has insufficient history after alignment")

    return data


def adv30(df: pd.DataFrame) -> pd.Series:
    return (df["close"] * df["volume"]).rolling(30, min_periods=30).mean()


def liquidity_mask(df: pd.DataFrame) -> pd.Series:
    a = adv30(df)
    d = df["close"] * df["volume"]
    return (a > MIN_DOLLAR_VOL_AVG_30D) & (d > MIN_DAILY_DOLLAR_VOL)

# ------------------------------ Indicators ------------------------------ #

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip)
    ref = past.shift(lookback)
    return (past / ref - 1.0)

def realized_vol(close: pd.Series, window: int) -> pd.Series:
    rets = close.pct_change()
    return rets.rolling(window, min_periods=window).std() * np.sqrt(252)

# --------------------------- Position Handling --------------------------- #

@dataclass
class Position:
    symbol: str
    qty: float
    entry_px: float

def round_qty(qty: float) -> float:
    if not np.isfinite(qty) or qty <= 0:
        return 0.0
    return float(np.floor(qty * (10 ** QTY_DECIMALS)) / (10 ** QTY_DECIMALS))

def apply_trade_cost(price: float, side: str) -> float:
    slip = price * (SLIPPAGE_BPS / 10_000)
    fee = price * (FEE_BPS / 10_000)
    return price + slip + fee if side == "buy" else price - slip - fee

# ------------------------------ Core Logic ------------------------------ #

def build_panel(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s: df["open"] for s, df in data.items()}, index=idx).reindex(close.index)
    volmask = pd.DataFrame({s: liquidity_mask(df) for s, df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp, volmask

def compute_momentum_scores(close: pd.DataFrame) -> pd.DataFrame:
    mom = trailing_return(close, LOOKBACK_DAYS, SKIP_RECENT_DAYS)
    return mom

def compute_inv_vol_weights(close: pd.DataFrame) -> pd.DataFrame:
    vol = close.apply(lambda s: realized_vol(s, VOL_LOOKBACK))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / vol.replace(0, np.nan)
    return inv

def monthly_rebalance_dates_from_close(close: pd.DataFrame) -> pd.DatetimeIndex:
    last_per_month = close.index.to_series().groupby(close.index.to_period("M")).max()
    return pd.DatetimeIndex(last_per_month.values)

def rebalance_weights(
        reb_date: pd.Timestamp,
        close: pd.DataFrame,
        volmask: pd.DataFrame,
        mom: pd.DataFrame,
        inv_vol: pd.DataFrame
) -> pd.Series:
    if reb_date not in close.index:
        raise KeyError("Rebalance date not in price index")

    eligible = []
    for s in close.columns:
        if s not in mom.columns:
            continue
        liq_ok = bool(volmask.loc[reb_date, s]) if s in volmask.columns else True
        mom_val = mom.loc[reb_date, s] if s in mom.columns else np.nan
        if liq_ok and np.isfinite(mom_val):
            eligible.append((s, float(mom_val)))

    if not eligible:
        return pd.Series(dtype=float)

    eligible.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in eligible[:N_POSITIONS]]

    raw = {}
    for s in top_symbols:
        iv = inv_vol.loc[reb_date, s] if s in inv_vol.columns else np.nan
        if not np.isfinite(iv) or iv <= 0:
            iv = 0.0
        raw[s] = iv

    if not raw:
        return pd.Series(dtype=float)

    raw_sum = sum(raw.values())
    if raw_sum <= 0:
        w = {s: 1.0 / len(raw) for s in raw.keys()}
    else:
        w = {s: raw[s] / raw_sum for s in raw.keys()}

    w = {s: min(MAX_WEIGHT_PER_ASSET, max(MIN_WEIGHT_PER_ASSET, wv)) for s, wv in w.items()}
    cap_sum = sum(w.values())
    if cap_sum > 0:
        w = {s: wv / cap_sum for s, wv in w.items()}

    return pd.Series(w, dtype=float)

def run_backtest(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx, close, openp, volmask = build_panel(data)
    mom = compute_momentum_scores(close)
    inv_vol = compute_inv_vol_weights(close)

    reb_days = monthly_rebalance_dates_from_close(close)
    earliest_allowed = idx[0] + pd.Timedelta(days=MIN_HISTORY_DAYS)
    reb_days = reb_days[reb_days >= earliest_allowed]

    equity = pd.Series(index=idx, dtype=float)
    exposures = pd.DataFrame(0.0, index=idx, columns=[c for c in close.columns if c != "CASH"], dtype=float)
    cash = START_CAPITAL
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, object]] = []

    def portfolio_value(dt: pd.Timestamp) -> float:
        val = cash
        for s, p in positions.items():
            if dt in close.index and s in close.columns:
                px = float(close.loc[dt, s])
                val += p.qty * px
        return val

    for i, dt in enumerate(idx):
        equity.iloc[i] = portfolio_value(dt)

        if dt in reb_days:
            target_w = rebalance_weights(dt, close, volmask, mom, inv_vol)

            if i + 1 < len(idx):
                t_exec = idx[i + 1]
                port_val_dt = portfolio_value(dt)

                target_notional = {}
                for s in target_w.index:
                    if s not in openp.columns or t_exec not in openp.index:
                        continue
                    target_notional[s] = float(target_w[s]) * port_val_dt

                current_notional = {}
                for s, p in positions.items():
                    if dt in close.index and s in close.columns:
                        current_notional[s] = p.qty * float(close.loc[dt, s])

                all_syms = sorted(set(list(current_notional.keys()) + list(target_notional.keys())))

                # Sells
                for s in all_syms:
                    open_px = float(openp.loc[t_exec, s]) if (s in openp.columns) else np.nan
                    if not np.isfinite(open_px) or open_px <= 0:
                        continue
                    tgt = target_notional.get(s, 0.0)
                    cur = current_notional.get(s, 0.0)
                    diff = tgt - cur
                    if diff < -1e-8 and s in positions:
                        qty = round_qty((-diff) / open_px)
                        if qty > 0:
                            exec_px = apply_trade_cost(open_px, "sell")
                            cash += qty * exec_px
                            positions[s].qty -= qty
                            trades.append({"date": t_exec, "symbol": s, "side": "SELL", "qty": qty, "price": exec_px, "reason": "rebalance"})
                            if positions[s].qty <= 0:
                                positions.pop(s, None)

                # Buys
                for s in all_syms:
                    open_px = float(openp.loc[t_exec, s]) if (s in openp.columns) else np.nan
                    if not np.isfinite(open_px) or open_px <= 0:
                        continue
                    tgt = target_notional.get(s, 0.0)
                    cur = 0.0
                    if s in positions and dt in close.index and s in close.columns:
                        cur = positions[s].qty * float(close.loc[dt, s])
                    diff = tgt - cur
                    if diff > 1e-8:
                        exec_px = apply_trade_cost(open_px, "buy")
                        qty = round_qty(diff / exec_px)
                        if qty > 0 and qty * exec_px <= cash:
                            cash -= qty * exec_px
                            if s in positions:
                                positions[s].qty += qty
                            else:
                                positions[s] = Position(symbol=s, qty=qty, entry_px=exec_px)
                            trades.append({"date": t_exec, "symbol": s, "side": "BUY", "qty": qty, "price": exec_px, "reason": "rebalance"})

        for s in exposures.columns:
            if s in positions and dt in close.index and s in close.columns:
                exposures.loc[dt, s] = positions[s].qty * float(close.loc[dt, s])
            else:
                exposures.loc[dt, s] = 0.0

    if len(idx) > 0:
        equity.iloc[-1] = portfolio_value(idx[-1])

    trades_df = pd.DataFrame(trades, columns=["date", "symbol", "side", "qty", "price", "reason"])
    trades_df.sort_values("date", inplace=True)

    equity_df = equity.to_frame("equity")
    exposures = exposures.fillna(0.0)

    return equity_df, exposures, trades_df

# -------------------------------- Benchmark ------------------------------ #

def compute_benchmark_equity(
        mode: str,
        data: Dict[str, pd.DataFrame],
        idx: pd.DatetimeIndex,
        close: Optional[pd.DataFrame] = None,
        mom: Optional[pd.DataFrame] = None
) -> pd.Series:
    mode = mode.upper().strip()
    if mode == "VOO":
        sym = "VOO"
        if sym not in data:
            raise ValueError("VOO data not available for benchmark")
        c = data[sym]["close"].reindex(idx).dropna()
        base = c.iloc[0]
        return (START_CAPITAL * (c / base)).reindex(idx).ffill()

    if mode == "QQQ":
        sym = "QQQ"
        if sym not in data:
            raise ValueError("QQQ data not available for benchmark")
        c = data[sym]["close"].reindex(idx).dropna()
        base = c.iloc[0]
        return (START_CAPITAL * (c / base)).reindex(idx).ffill()

    if mode == "EQUAL":
        if close is None:
            close = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
        valid_cols = [c for c in close.columns if close[c].notna().sum() == len(close)]
        if not valid_cols:
            raise ValueError("No fully aligned series for equal-weight benchmark")
        norm = close[valid_cols] / close[valid_cols].iloc[0]
        eq = START_CAPITAL * norm.mean(axis=1)
        return eq.reindex(idx).ffill()

    if mode == "TOP5_STATIC":
        if close is None:
            close = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
        if mom is None:
            mom = trailing_return(close, LOOKBACK_DAYS, SKIP_RECENT_DAYS)
        # pick the first date where momentum exists
        first_valid = mom.dropna(how="all").index.min()
        if pd.isna(first_valid):
            raise ValueError("Insufficient history to form TOP5_STATIC benchmark")
        m = mom.loc[first_valid].dropna().sort_values(ascending=False)
        top = list(m.index[:N_POSITIONS])
        sub = close[top].dropna()
        if len(sub) == 0:
            raise ValueError("No valid series for TOP5_STATIC benchmark")
        weights = np.repeat(1.0 / len(top), len(top))
        base = sub.iloc[0]
        rel = sub.divide(base)
        eq = START_CAPITAL * (rel @ weights)
        return eq.reindex(idx).ffill()

    raise ValueError(f"Unknown BENCHMARK_MODE: {mode}")

# -------------------------------- Metrics -------------------------------- #

def compute_stats(equity: pd.Series) -> Dict[str, float]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"CAGR": np.nan, "TotalReturn": np.nan, "MaxDD": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MAR": np.nan}
    rets = eq.pct_change().iloc[1:]
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total_return) ** (1 / years) - 1 if years and years > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0)
    max_dd = float(dd.min()) if len(dd) else np.nan
    vol = float(rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    mar = float(cagr / abs(max_dd)) if (isinstance(max_dd, float) and max_dd < 0 and cagr is not None and not np.isnan(cagr)) else np.nan
    return {"CAGR": cagr, "TotalReturn": total_return, "MaxDD": max_dd, "Vol": vol, "Sharpe": sharpe, "MAR": mar}

# ------------------------------- Manifest -------------------------------- #

def _config_dict() -> Dict[str, object]:
    return {
        "SYMBOLS": SYMBOLS,
        "BASE_SYMBOL": BASE_SYMBOL,
        "YF_PERIOD": YF_PERIOD,
        "START_CAPITAL": START_CAPITAL,
        "N_POSITIONS": N_POSITIONS,
        "LOOKBACK_DAYS": LOOKBACK_DAYS,
        "SKIP_RECENT_DAYS": SKIP_RECENT_DAYS,
        "VOL_LOOKBACK": VOL_LOOKBACK,
        "MAX_WEIGHT_PER_ASSET": MAX_WEIGHT_PER_ASSET,
        "SLIPPAGE_BPS": SLIPPAGE_BPS,
        "FEE_BPS": FEE_BPS,
        "MIN_DOLLAR_VOL_AVG_30D": MIN_DOLLAR_VOL_AVG_30D,
        "MIN_DAILY_DOLLAR_VOL": MIN_DAILY_DOLLAR_VOL,
        "QTY_DECIMALS": QTY_DECIMALS,
        "BENCHMARK_MODE": BENCHMARK_MODE,
    }

def _write_run_manifest(stats: Dict[str, float], bench_stats: Dict[str, float], syms: List[str]) -> None:
    cfg = _config_dict()
    cfg_json = json.dumps(cfg, sort_keys=True).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_json).hexdigest()
    manifest = {
        "config": cfg,
        "config_hash": cfg_hash,
        "stats_strategy": stats,
        "stats_benchmark": bench_stats,
        "universe": syms
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/run.json", "w") as f:
        json.dump(manifest, f, indent=2)

# --------------------------------- Main ---------------------------------- #

def main():
    data = fetch_data(SYMBOLS, period=YF_PERIOD)

    # Strategy
    equity_df, expo_df, trades_df = run_backtest(data)
    stats = compute_stats(equity_df["equity"])

    # Build aligned price panel for benchmark construction
    idx = equity_df.index
    close_panel = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
    mom_panel = trailing_return(close_panel, LOOKBACK_DAYS, SKIP_RECENT_DAYS)

    # Benchmark
    bench_series = compute_benchmark_equity(BENCHMARK_MODE, data, idx, close_panel, mom_panel)
    equity_df["benchmark"] = bench_series

    bench_stats = compute_stats(equity_df["benchmark"])

    # Output
    print("Backtest Stats (Strategy):")
    for k, v in stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")

    print(f"\nBenchmark ({BENCHMARK_MODE}) Stats:")
    for k, v in bench_stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")

    os.makedirs(OUT_DIR, exist_ok=True)
    equity_df.to_csv(f"{OUT_DIR}/equity_curve.csv")            # now includes 'equity' and 'benchmark'
    expo_df.to_csv(f"{OUT_DIR}/exposures.csv")
    trades_df.to_csv(f"{OUT_DIR}/trades.csv", index=False)

    compare_df = pd.DataFrame({
        "strategy": equity_df["equity"],
        "benchmark": equity_df["benchmark"]
    }, index=equity_df.index)
    compare_df.to_csv(f"{OUT_DIR}/compare_equity.csv")

    _write_run_manifest(stats, bench_stats, SYMBOLS)
    print(f"\nSaved: {OUT_DIR}/equity_curve.csv, {OUT_DIR}/compare_equity.csv, {OUT_DIR}/exposures.csv, {OUT_DIR}/trades.csv, {OUT_DIR}/run.json")

    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        equity_df[["equity", "benchmark"]].plot()
        plt.title(f"Strategy vs Benchmark ({BENCHMARK_MODE})")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        fp = f"{OUT_DIR}/equity_curve.png"
        plt.savefig(fp)
        plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__ == "__main__":
    main()
