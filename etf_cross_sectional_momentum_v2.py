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

# ================================ Config ================================= #

# Universe (BASE) + optional EXTENDED set. Duplicates are auto-removed.
UNIVERSE_BASE: List[str] = [
    "VOO","QQQ","IWM","DIA","XLK","XLF","XLV","XLY","XLP","XLI","XLU","XLE","XLRE","XLB"
]
UNIVERSE_EXT: List[str] = [
    "SMH","SOXX","IGV","FDN","XBI","IBB","XLC","ICLN","TAN","PBW","ARKK"
]

BASE_SYMBOL = "VOO"                  # benchmark symbol and calendar anchor
YF_PERIOD = "max"

START_CAPITAL = 100_000.0

# ---- Optimizer-equivalent knobs ----
UNIVERSE_MODE = "EXT"                # "BASE" or "EXT"
N_POSITIONS = 1                      # 1..N
LOOKBACKS: Tuple[int,int,int] = (252, 126, 21)  # composite momentum windows
LOOKBACK_WEIGHTS: Tuple[float,float,float] = (0.4, 0.4, 0.2)  # must sum ~1.0
SKIP_RECENT_DAYS = 0                 # 0 or 21 (12-1)
REBALANCE_FREQ = "2W"                # "W" | "2W" | "M"
INV_VOL = False                      # inverse-volatility scaling on/off
VOL_LOOKBACK = 20                    # realized vol window
MAX_WEIGHT_PER_ASSET = 0.60          # cap per asset
ABSOLUTE_MOMENTUM = True             # gate: score must be > 0
MA_FILTER = False                    # gate: price > MA200
MA_WINDOW = 200

# Costs (per-side)
SLIPPAGE_BPS = 1
FEE_BPS = 1
QTY_DECIMALS = 4

# Liquidity guards (conservative for ETFs)
MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000

# Benchmark: "VOO" | "QQQ" | "EQUAL" | "TOP5_STATIC"
BENCHMARK_MODE = "VOO"

OUT_DIR = "./out"

# ============================== Data Layer =============================== #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    symbols = _dedupe(symbols)
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
                continue
            data[s] = clean_df(raw[s].copy())
    else:
        s = symbols[0]
        data[s] = clean_df(raw)

    if BASE_SYMBOL not in data:
        raise ValueError(f"BASE_SYMBOL {BASE_SYMBOL} missing in Yahoo response")

    # Align all series to BASE_SYMBOL calendar and drop NaNs
    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

    # Liquidity mask; drop series illiquid too often
    def liq_ok(df: pd.DataFrame) -> pd.Series:
        adv = (df["close"] * df["volume"]).rolling(30, min_periods=30).mean()
        daily = df["close"] * df["volume"]
        return (adv > MIN_DOLLAR_VOL_AVG_30D) & (daily > MIN_DAILY_DOLLAR_VOL)

    keep: Dict[str, pd.DataFrame] = {}
    for s, df in data.items():
        m = liq_ok(df)
        if m.sum() >= int(0.6 * len(m)):  # at least 60% of days pass
            keep[s] = df
    if BASE_SYMBOL not in keep:
        keep[BASE_SYMBOL] = data[BASE_SYMBOL]
    return keep

# ============================== Indicators =============================== #

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip)
    ref = past.shift(lookback)
    return past.divide(ref) - 1.0

def realized_vol(close: pd.Series, window: int) -> pd.Series:
    r = close.pct_change()
    return r.rolling(window, min_periods=window).std() * np.sqrt(252)

def composite_momentum(close: pd.DataFrame,
                       lookbacks: Tuple[int,int,int],
                       weights: Tuple[float,float,float],
                       skip: int) -> pd.DataFrame:
    l1, l2, l3 = lookbacks
    w1, w2, w3 = weights
    m1 = trailing_return(close, l1, skip)
    m2 = trailing_return(close, l2, skip)
    m3 = trailing_return(close, l3, skip)
    return w1 * m1 + w2 * m2 + w3 * m3

def inverse_volatility_matrix(close: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = close.apply(lambda s: realized_vol(s, window))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / vol.replace(0, np.nan)
    return inv

# ============================== Execution ================================ #

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

def build_panel(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s: df["open"] for s, df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f = freq.upper()
    if f == "M":
        g = idx.to_series().groupby(idx.to_period("M")).max()
    elif f == "W":
        g = idx.to_series().groupby(idx.to_period("W")).max()
    elif f == "2W":
        weekly_last = idx.to_series().groupby(idx.to_period("W")).max()
        g = weekly_last.iloc[::2]
    else:
        raise ValueError("REBALANCE_FREQ must be 'W','2W','M'")
    return pd.DatetimeIndex(g.values)

def compute_weights(
        dt: pd.Timestamp,
        close: pd.DataFrame,
        mom: pd.DataFrame,
        inv_vol: Optional[pd.DataFrame],
        max_w: float,
        absolute_mom: bool,
        ma_filter: bool,
        ma200: Optional[pd.DataFrame],
        n_positions: int
) -> pd.Series:
    # Eligibility gate
    eligible: List[Tuple[str, float]] = []
    for s in close.columns:
        sc = float(mom.loc[dt, s]) if s in mom.columns else np.nan
        if not np.isfinite(sc):
            continue
        if absolute_mom and sc <= 0:
            continue
        if ma_filter:
            m = float(ma200.loc[dt, s]) if (ma200 is not None and s in ma200.columns) else np.nan
            if not np.isfinite(m) or float(close.loc[dt, s]) <= m:
                continue
        eligible.append((s, sc))

    if not eligible:
        return pd.Series(dtype=float)

    eligible.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in eligible[:n_positions]]

    # Base raw weights: momentum * inverse-vol (if enabled)
    raw: Dict[str, float] = {}
    for s in top:
        sc = float(mom.loc[dt, s])
        iv = float(inv_vol.loc[dt, s]) if inv_vol is not None and s in inv_vol.columns else 1.0
        iv = max(iv, 0.0)
        sc = max(sc, 0.0)
        raw[s] = sc * (iv if INV_VOL else 1.0)

    if sum(raw.values()) <= 0:
        raw = {s: 1.0 for s in top}

    w = {s: v / sum(raw.values()) for s, v in raw.items()}
    w = {s: min(max_w, max(0.0, wv)) for s, wv in w.items()}
    ssum = sum(w.values())
    w = {s: (wv / ssum) for s, wv in w.items()} if ssum > 0 else {s: 1.0/len(top) for s in top}
    return pd.Series(w, dtype=float)

def run_backtest(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Universe pick
    if UNIVERSE_MODE.upper() == "BASE":
        syms = _dedupe(UNIVERSE_BASE)
    else:
        syms = _dedupe(UNIVERSE_BASE + UNIVERSE_EXT)
    syms = [s for s in syms if s in data]
    if BASE_SYMBOL not in syms:
        syms = [BASE_SYMBOL] + syms
    data_u = {s: data[s] for s in syms}

    idx, close, openp = build_panel(data_u)

    mom = composite_momentum(close, LOOKBACKS, LOOKBACK_WEIGHTS, SKIP_RECENT_DAYS)
    inv = inverse_volatility_matrix(close, VOL_LOOKBACK) if INV_VOL else None
    ma200 = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean() if MA_FILTER else None

    # Earliest date with enough history for largest lookback
    earliest = idx[0] + pd.Timedelta(days=max(LOOKBACKS) + SKIP_RECENT_DAYS + 10)
    reb_days = last_day_per_period(idx, REBALANCE_FREQ)
    reb_days = reb_days[reb_days >= earliest]

    equity = pd.Series(index=idx, dtype=float)
    exposures = pd.DataFrame(0.0, index=idx, columns=[c for c in close.columns if c != "CASH"], dtype=float)
    trades: List[Dict[str, object]] = []

    cash = START_CAPITAL
    positions: Dict[str, Position] = {}

    def portfolio_value(t: pd.Timestamp) -> float:
        val = cash
        for s, pos in positions.items():
            if s in close.columns:
                px = float(close.loc[t, s])
                val += pos.qty * px
        return val

    for i, dt in enumerate(idx):
        equity.iloc[i] = portfolio_value(dt)

        if dt in reb_days and i + 1 < len(idx):
            t_exec = idx[i + 1]
            port_val = portfolio_value(dt)
            tgt_w = compute_weights(
                dt, close, mom, inv, MAX_WEIGHT_PER_ASSET,
                ABSOLUTE_MOMENTUM, MA_FILTER, ma200, N_POSITIONS
            )

            if tgt_w.empty:
                # go to cash
                for s, p in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and p.qty > 0:
                        px = apply_trade_cost(opx, "sell")
                        cash += p.qty * px
                        trades.append({"date": t_exec, "symbol": s, "side": "SELL", "qty": p.qty, "price": px, "reason": "rebalance_clear"})
                positions.clear()
            else:
                target_notional = {s: float(tgt_w[s]) * port_val for s in tgt_w.index}
                cur_notional = {s: (pos.qty * float(close.loc[dt, s])) for s, pos in positions.items()}
                all_syms = sorted(set(cur_notional.keys()).union(target_notional.keys()))

                # Sells first
                for s in all_syms:
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if not np.isfinite(opx): continue
                    tgt = target_notional.get(s, 0.0)
                    cur = cur_notional.get(s, 0.0)
                    diff = tgt - cur
                    if diff < -1e-8 and s in positions:
                        qty = round_qty((-diff) / opx)
                        if qty > 0:
                            px = apply_trade_cost(opx, "sell")
                            cash += qty * px
                            positions[s].qty -= qty
                            trades.append({"date": t_exec, "symbol": s, "side": "SELL", "qty": qty, "price": px, "reason": "rebalance"})
                            if positions[s].qty <= 0:
                                positions.pop(s, None)

                # Buys
                for s in all_syms:
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if not np.isfinite(opx): continue
                    tgt = target_notional.get(s, 0.0)
                    cur = cur_notional.get(s, 0.0)
                    diff = tgt - cur
                    if diff > 1e-8 and cash > 0:
                        px = apply_trade_cost(opx, "buy")
                        qty = round_qty(min(diff, cash) / px)
                        if qty > 0:
                            cost = qty * px
                            cash -= cost
                            if s in positions:
                                positions[s].qty += qty
                            else:
                                positions[s] = Position(symbol=s, qty=qty, entry_px=px)
                            trades.append({"date": t_exec, "symbol": s, "side": "BUY", "qty": qty, "price": px, "reason": "rebalance"})

        # Exposures at close
        for s in exposures.columns:
            if s in positions:
                exposures.loc[dt, s] = positions[s].qty * float(close.loc[dt, s])
            else:
                exposures.loc[dt, s] = 0.0

    if len(idx) > 0:
        equity.iloc[-1] = portfolio_value(idx[-1])

    trades_df = pd.DataFrame(trades, columns=["date", "symbol", "side", "qty", "price", "reason"]).sort_values("date")
    equity_df = equity.to_frame("equity")
    exposures = exposures.fillna(0.0)
    return equity_df, exposures, trades_df

# ============================== Benchmark ================================ #

def compute_benchmark_equity(
        mode: str,
        data: Dict[str, pd.DataFrame],
        idx: pd.DatetimeIndex,
        close: Optional[pd.DataFrame] = None,
        mom: Optional[pd.DataFrame] = None
) -> pd.Series:
    m = mode.upper().strip()
    if m in ("VOO","QQQ"):
        sym = m
        if sym not in data:
            raise ValueError(f"{sym} data unavailable for benchmark")
        c = data[sym]["close"].reindex(idx).dropna()
        base = c.iloc[0]
        return (START_CAPITAL * (c / base)).reindex(idx).ffill()

    if m == "EQUAL":
        if close is None:
            raise ValueError("close panel required for EQUAL benchmark")
        valid = [c for c in close.columns if close[c].notna().sum() == len(close)]
        if not valid:
            raise ValueError("No fully aligned series for equal-weight benchmark")
        norm = close[valid] / close[valid].iloc[0]
        eq = START_CAPITAL * norm.mean(axis=1)
        return eq.reindex(idx).ffill()

    if m == "TOP5_STATIC":
        if close is None:
            raise ValueError("close panel required for TOP5_STATIC")
        if mom is None:
            raise ValueError("momentum panel required for TOP5_STATIC")
        first_valid = mom.dropna(how="all").index.min()
        if pd.isna(first_valid):
            raise ValueError("Insufficient history for TOP5_STATIC")
        mser = mom.loc[first_valid].dropna().sort_values(ascending=False)
        top = list(mser.index[:max(1, N_POSITIONS)])
        sub = close[top].dropna()
        if len(sub) == 0:
            raise ValueError("No valid series for TOP5_STATIC")
        weights = np.repeat(1.0 / len(top), len(top))
        base = sub.iloc[0]
        rel = sub.divide(base)
        eq = START_CAPITAL * (rel @ weights)
        return eq.reindex(idx).ffill()

    raise ValueError(f"Unknown BENCHMARK_MODE: {mode}")

# ================================ Metrics ================================= #

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

# ================================ Manifest ================================= #

def _config_dict() -> Dict[str, object]:
    return {
        "UNIVERSE_BASE": UNIVERSE_BASE,
        "UNIVERSE_EXT": UNIVERSE_EXT,
        "BASE_SYMBOL": BASE_SYMBOL,
        "YF_PERIOD": YF_PERIOD,
        "START_CAPITAL": START_CAPITAL,
        "UNIVERSE_MODE": UNIVERSE_MODE,
        "N_POSITIONS": N_POSITIONS,
        "LOOKBACKS": LOOKBACKS,
        "LOOKBACK_WEIGHTS": LOOKBACK_WEIGHTS,
        "SKIP_RECENT_DAYS": SKIP_RECENT_DAYS,
        "REBALANCE_FREQ": REBALANCE_FREQ,
        "INV_VOL": INV_VOL,
        "VOL_LOOKBACK": VOL_LOOKBACK,
        "MAX_WEIGHT_PER_ASSET": MAX_WEIGHT_PER_ASSET,
        "ABSOLUTE_MOMENTUM": ABSOLUTE_MOMENTUM,
        "MA_FILTER": MA_FILTER,
        "MA_WINDOW": MA_WINDOW,
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

# ================================= Main ================================== #

def main():
    # Pick universe
    symbols = _dedupe(UNIVERSE_BASE if UNIVERSE_MODE.upper()=="BASE" else (UNIVERSE_BASE + UNIVERSE_EXT))
    data = fetch_data(symbols)

    equity_df, expo_df, trades_df = run_backtest(data)
    stats = compute_stats(equity_df["equity"])

    # Build aligned panels for benchmark
    idx = equity_df.index
    close_panel = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
    mom_panel = composite_momentum(close_panel, LOOKBACKS, LOOKBACK_WEIGHTS, SKIP_RECENT_DAYS)

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
    equity_df.to_csv(f"{OUT_DIR}/equity_curve.csv")
    expo_df.to_csv(f"{OUT_DIR}/exposures.csv")
    trades_df.to_csv(f"{OUT_DIR}/trades.csv", index=False)

    compare_df = pd.DataFrame({"strategy": equity_df["equity"], "benchmark": equity_df["benchmark"]}, index=equity_df.index)
    compare_df.to_csv(f"{OUT_DIR}/compare_equity.csv")

    _write_run_manifest(stats, bench_stats, list(data.keys()))
    print(f"\nSaved: {OUT_DIR}/equity_curve.csv, {OUT_DIR}/compare_equity.csv, {OUT_DIR}/exposures.csv, {OUT_DIR}/trades.csv, {OUT_DIR}/run.json")

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
