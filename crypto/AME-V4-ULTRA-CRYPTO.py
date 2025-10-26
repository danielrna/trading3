# AME-V4-ULTRA-CRYPTO Backtest (standalone)
# Objective: MAX CAGR-style backtest using ULTRA params from optimizer runs.
# Spot-only, no leverage/derivatives executed, but concentration is allowed via weight caps.
# Python 3.10+
# Deps: pandas, numpy, yfinance, matplotlib (optional)

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

BASE_SYMBOL = "BTC-USD"                # benchmark & calendar anchor
YF_PERIOD = "max"
START_CAPITAL = 1000.0
OUT_DIR = "./out"

# Crypto spot universe (USD pairs)
UNIVERSE: List[str] = [
    # majors
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "AVAX-USD", "MATIC-USD", "DOT-USD",
    "LTC-USD", "BCH-USD", "LINK-USD", "ATOM-USD", "UNI-USD",
    "XLM-USD", "ETC-USD",
    # defensives / stables (kept in panel, excluded from selection)
    "USDT-USD", "USDC-USD",
]

# -------- ULTRA parameters (fixed from your run: Gen ~7–11 best) -------- #

LOOKBACKS: Tuple[int, int, int] = (7, 3, 1)
LOOKBACK_WEIGHTS: Tuple[float, float, float] = (0.5, 0.3, 0.2)
SKIP_RECENT_DAYS: int = 0

REBALANCE_FREQ: str = "D"             # Daily
ABS_MOM_GATE_BASE: float = 0.0        # Force engagement; if none > gate, pick top-1 anyway
SPREAD_THRESHOLD1: float = 0.001      # 0.10%
SPREAD_THRESHOLD2: float = 0.005      # 0.50%
ALLOW_TWO_WHEN_SPREAD: bool = True
ALLOW_THREE_WHEN_SPREAD: bool = True

# Dynamic sizing via spread (pseudo-concentration)
MAX_WEIGHT_MIN: float = 1.00
MAX_WEIGHT_MAX: float = 2.50
CASH_BUFFER_MIN: float = 0.00
CASH_BUFFER_MAX: float = 0.02

# No defensive override in ULTRA; if no eligible above gate, still pick top-1 by momentum
DEFENSIVE_OVERRIDE: bool = False
DEFENSIVE_ASSETS = {"USDT-USD", "USDC-USD"}

# Trading costs (per-side, conservative for spot)
SLIPPAGE_BPS = 10
FEE_BPS = 10
QTY_DECIMALS = 8

# Liquidity guards (soft; still keep BTC calendar alignment)
MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000

# Benchmark: BASE buy&hold
BENCHMARK_MODE = "BASE"               # BASE = BTC-USD

# ============================== Utilities ================================ #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    symbols = _dedupe(symbols)
    raw = yf.download(
        symbols, period=period, interval="1d",
        group_by="ticker", auto_adjust=False, threads=True, progress=False
    )
    data: Dict[str, pd.DataFrame] = {}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        d = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume"
        }).copy()
        d = d.dropna(subset=["open", "high", "low", "close", "volume"])
        bad = (d["open"] <= 0) | (d["high"] <= 0) | (d["low"] <= 0) | (d["close"] <= 0) | (d["volume"] < 0)
        d = d[~bad]
        d.index.name = "date"
        return d[["open", "high", "low", "close", "volume"]]

    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        for s in symbols:
            if s in top:
                data[s] = clean(raw[s].copy())
    else:
        data[symbols[0]] = clean(raw)

    if BASE_SYMBOL not in data:
        raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo Finance")

    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

    # Soft liquidity screen
    def liq_ok(df: pd.DataFrame) -> pd.Series:
        adv = (df["close"] * df["volume"]).rolling(30, min_periods=30).mean()
        daily = df["close"] * df["volume"]
        return (adv > MIN_DOLLAR_VOL_AVG_30D) & (daily > MIN_DAILY_DOLLAR_VOL)

    keep: Dict[str, pd.DataFrame] = {}
    for s, df in data.items():
        m = liq_ok(df)
        if m.sum() >= int(0.6 * len(m)):
            keep[s] = df
    if BASE_SYMBOL not in keep:
        keep[BASE_SYMBOL] = data[BASE_SYMBOL]
    return keep

def round_qty(q: float) -> float:
    if not np.isfinite(q) or q <= 0:
        return 0.0
    return float(np.floor(q * (10 ** QTY_DECIMALS)) / (10 ** QTY_DECIMALS))

def apply_trade_cost(price: float, side: str) -> float:
    slip = price * (SLIPPAGE_BPS / 10_000)
    fee = price * (FEE_BPS / 10_000)
    return price + slip + fee if side == "buy" else price - slip - fee

def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f = (freq or "D").upper()
    s = idx.to_series()
    if f == "M":
        g = s.groupby(idx.to_period("M")).max()
    elif f == "W":
        g = s.groupby(idx.to_period("W")).max()
    elif f == "2W":
        weekly_last = s.groupby(idx.to_period("W")).max()
        g = weekly_last.iloc[::2]
    elif f == "D":
        g = s
    else:
        raise ValueError("REBALANCE_FREQ must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)

# ============================== Indicators =============================== #

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip)
    ref = past.shift(lookback)
    return past.divide(ref) - 1.0

def composite_momentum(close: pd.DataFrame,
                       lookbacks: Tuple[int, int, int],
                       weights: Tuple[float, float, float],
                       skip: int) -> pd.DataFrame:
    l1, l2, l3 = lookbacks
    w1, w2, w3 = weights
    m1 = trailing_return(close, l1, skip)
    m2 = trailing_return(close, l2, skip)
    m3 = trailing_return(close, l3, skip)
    return w1 * m1 + w2 * m2 + w3 * m3

# ============================== Execution ================================ #

@dataclass
class Position:
    symbol: str
    qty: float
    entry_px: float

def build_panel(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s: df["open"] for s, df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def dynamic_caps(spread_top12: float) -> Tuple[float, float]:
    """
    Map spread in [0,10%] to:
      - weight cap in [MAX_WEIGHT_MIN, MAX_WEIGHT_MAX]
      - cash buffer in [CASH_BUFFER_MAX -> CASH_BUFFER_MIN]
    """
    sp = max(0.0, min(0.10, float(spread_top12)))
    frac = sp / 0.10
    cap = MAX_WEIGHT_MIN + (MAX_WEIGHT_MAX - MAX_WEIGHT_MIN) * frac
    cash_buf = CASH_BUFFER_MAX - (CASH_BUFFER_MAX - CASH_BUFFER_MIN) * frac
    return min(MAX_WEIGHT_MAX, cap), max(0.0, min(0.50, cash_buf))

def compute_target_weights(
        dt: pd.Timestamp,
        close: pd.DataFrame,
        mom: pd.DataFrame
) -> Tuple[Dict[str, float], float]:
    """
    Returns target weights and cash buffer:
      - Excludes defensives from selection
      - Uses absolute momentum gate; if none pass, still pick top-1 by momentum
      - Spread-based enabling of up to 3 assets and dynamic caps/cash buffer
    """
    scores = mom.loc[dt].dropna().copy()
    # exclude defensives
    scores = scores[[s for s in scores.index if s not in DEFENSIVE_ASSETS and s in close.columns]]

    # Eligible above gate
    elig = [(s, float(sc)) for s, sc in scores.items() if np.isfinite(sc) and sc > ABS_MOM_GATE_BASE]
    if not elig:
        # Force top-1 by raw momentum even if negative (ULTRA behavior)
        if len(scores) == 0:
            return {}, CASH_BUFFER_MAX
        top_sym = scores.sort_values(ascending=False).index.tolist()[0]
        return {top_sym: 1.0 - CASH_BUFFER_MAX}, CASH_BUFFER_MAX

    elig.sort(key=lambda x: x[1], reverse=True)
    chosen: List[str] = [elig[0][0]]

    spread1 = elig[0][1] - (elig[1][1] if len(elig) >= 2 else -1e9)
    spread2 = (elig[1][1] - elig[2][1]) if len(elig) >= 3 else -1e9

    if ALLOW_TWO_WHEN_SPREAD and len(elig) >= 2 and spread1 >= SPREAD_THRESHOLD1:
        chosen.append(elig[1][0])
    if ALLOW_THREE_WHEN_SPREAD and len(elig) >= 3 and spread2 >= SPREAD_THRESHOLD2:
        chosen.append(elig[2][0])

    cap, cash_buf = dynamic_caps(spread1 if np.isfinite(spread1) else 0.0)

    if len(chosen) == 1:
        target_w = {chosen[0]: min(cap, 1.0 - cash_buf)}
    elif len(chosen) == 2:
        raw = {chosen[0]: 1.0, chosen[1]: 0.9}
        ssum = sum(raw.values())
        raw = {k: v / ssum for k, v in raw.items()}
        capped = {k: min(cap, v) for k, v in raw.items()}
        ssum = sum(capped.values())
        target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {
            chosen[0]: 1.0 - cash_buf
        }
    else:
        raw = {chosen[0]: 1.0, chosen[1]: 0.9, chosen[2]: 0.8}
        ssum = sum(raw.values())
        raw = {k: v / ssum for k, v in raw.items()}
        capped = {k: min(cap, v) for k, v in raw.items()}
        ssum = sum(capped.values())
        target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {
            chosen[0]: 1.0 - cash_buf
        }

    return target_w, cash_buf

def run_backtest(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    syms = [s for s in _dedupe(UNIVERSE) if s in data]
    if BASE_SYMBOL not in syms:
        syms = [BASE_SYMBOL] + syms
    data_u = {s: data[s] for s in syms}

    idx, close, openp = build_panel(data_u)
    mom = composite_momentum(close, LOOKBACKS, LOOKBACK_WEIGHTS, SKIP_RECENT_DAYS)

    earliest = idx[0] + pd.Timedelta(days=max(LOOKBACKS) + SKIP_RECENT_DAYS + 10)
    reb_days = last_day_per_period(idx, REBALANCE_FREQ)
    reb_days = reb_days[reb_days >= earliest]

    equity = pd.Series(index=idx, dtype=float)
    exposures = pd.DataFrame(0.0, index=idx, columns=[c for c in close.columns if c != "CASH"], dtype=float)
    trades: List[Dict[str, object]] = []

    cash = START_CAPITAL
    positions: Dict[str, Position] = {}

    def pv(t: pd.Timestamp) -> float:
        v = cash
        for s, p in positions.items():
            if s in close.columns:
                v += p.qty * float(close.loc[t, s])
        return v

    for i, dt in enumerate(idx):
        equity.iloc[i] = pv(dt)

        if dt in reb_days and i + 1 < len(idx):
            t_exec = idx[i + 1]
            port_val = pv(dt)

            target_w, cash_buf = compute_target_weights(dt, close, mom)

            if not target_w:
                if DEFENSIVE_OVERRIDE:
                    # Not used in ULTRA config; kept for completeness
                    pass
                else:
                    # Liquidate to cash
                    for s, p in list(positions.items()):
                        opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                        if np.isfinite(opx) and p.qty > 0:
                            px = apply_trade_cost(opx, "sell")
                            cash += p.qty * px
                            trades.append({"date": t_exec, "symbol": s, "side": "SELL", "qty": p.qty, "price": px, "reason": "rebalance_clear"})
                    positions.clear()
            else:
                target_notional = {s: float(w) * port_val for s, w in target_w.items()}
                cur_notional = {s: (pos.qty * float(close.loc[dt, s])) for s, pos in positions.items()}
                all_syms = sorted(set(cur_notional.keys()).union(target_notional.keys()))

                # Sells
                for s in all_syms:
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if not np.isfinite(opx):
                        continue
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
                    if not np.isfinite(opx):
                        continue
                    tgt = target_notional.get(s, 0.0)
                    cur = cur_notional.get(s, 0.0)
                    diff = tgt - cur
                    if diff > 1e-8 and cash > 0:
                        px = apply_trade_cost(opx, "buy")
                        qty = round_qty(min(diff, cash) / px)
                        if qty > 0:
                            cost = qty * px
                            if cost <= cash:
                                cash -= cost
                                if s in positions:
                                    positions[s].qty += qty
                                else:
                                    positions[s] = Position(symbol=s, qty=qty, entry_px=px)
                                trades.append({"date": t_exec, "symbol": s, "side": "BUY", "qty": qty, "price": px, "reason": "rebalance"})

        # Exposures
        for s in exposures.columns:
            if s in positions:
                exposures.loc[dt, s] = positions[s].qty * float(close.loc[dt, s])
            else:
                exposures.loc[dt, s] = 0.0

    if len(idx) > 0:
        equity.iloc[-1] = pv(idx[-1])

    trades_df = pd.DataFrame(trades, columns=["date", "symbol", "side", "qty", "price", "reason"]).sort_values("date")
    equity_df = equity.to_frame("equity")
    exposures = exposures.fillna(0.0)
    return equity_df, exposures, trades_df, close

# ============================== Benchmark ================================ #

def compute_benchmark_equity(
        mode: str,
        data: Dict[str, pd.DataFrame],
        idx: pd.DatetimeIndex
) -> pd.Series:
    m = mode.upper().strip()
    if m == "BASE":
        if BASE_SYMBOL not in data:
            raise ValueError(f"{BASE_SYMBOL} missing for benchmark")
        c = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
        base = c.iloc[0]
        return (START_CAPITAL * (c / base)).reindex(idx).ffill()
    raise ValueError(f"Unknown BENCHMARK_MODE: {mode}")

# ================================ Metrics ================================ #

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
    vol = float(rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan
    mar = float(cagr / abs(max_dd)) if (isinstance(max_dd, float) and max_dd < 0 and cagr is not None and not np.isnan(cagr)) else np.nan
    return {"CAGR": cagr, "TotalReturn": total_return, "MaxDD": max_dd, "Vol": vol, "Sharpe": sharpe, "MAR": mar}

# ================================ Manifest =============================== #

def _config_dict() -> Dict[str, object]:
    return {
        "BASE_SYMBOL": BASE_SYMBOL,
        "YF_PERIOD": YF_PERIOD,
        "START_CAPITAL": START_CAPITAL,
        "UNIVERSE": UNIVERSE,
        "LOOKBACKS": LOOKBACKS,
        "LOOKBACK_WEIGHTS": LOOKBACK_WEIGHTS,
        "SKIP_RECENT_DAYS": SKIP_RECENT_DAYS,
        "REBALANCE_FREQ": REBALANCE_FREQ,
        "ABS_MOM_GATE_BASE": ABS_MOM_GATE_BASE,
        "SPREAD_THRESHOLD1": SPREAD_THRESHOLD1,
        "SPREAD_THRESHOLD2": SPREAD_THRESHOLD2,
        "ALLOW_TWO_WHEN_SPREAD": ALLOW_TWO_WHEN_SPREAD,
        "ALLOW_THREE_WHEN_SPREAD": ALLOW_THREE_WHEN_SPREAD,
        "MAX_WEIGHT_MIN": MAX_WEIGHT_MIN,
        "MAX_WEIGHT_MAX": MAX_WEIGHT_MAX,
        "CASH_BUFFER_MIN": CASH_BUFFER_MIN,
        "CASH_BUFFER_MAX": CASH_BUFFER_MAX,
        "DEFENSIVE_OVERRIDE": DEFENSIVE_OVERRIDE,
        "DEFENSIVE_ASSETS": list(DEFENSIVE_ASSETS),
        "SLIPPAGE_BPS": SLIPPAGE_BPS,
        "FEE_BPS": FEE_BPS,
        "QTY_DECIMALS": QTY_DECIMALS,
        "MIN_DOLLAR_VOL_AVG_30D": MIN_DOLLAR_VOL_AVG_30D,
        "MIN_DAILY_DOLLAR_VOL": MIN_DAILY_DOLLAR_VOL,
        "BENCHMARK_MODE": BENCHMARK_MODE,
    }

def _write_run_manifest(stats: Dict[str, float], bench_stats: Dict[str, float], syms: List[str]) -> None:
    cfg = _config_dict()
    manifest = {
        "config": cfg,
        "stats_strategy": stats,
        "stats_benchmark": bench_stats,
        "universe": syms
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/run_crypto_ultra.json", "w") as f:
        json.dump(manifest, f, indent=2)

# ================================= Main ================================== #

def main():
    symbols = _dedupe(UNIVERSE)
    print(f"Downloading {len(symbols)} tickers...")
    data = fetch_data(symbols)
    print(f"Universe after hygiene: {len(data)} tickers")

    equity_df, expo_df, trades_df, close_panel = run_backtest(data)
    stats = compute_stats(equity_df["equity"])

    idx = equity_df.index
    bench_series = compute_benchmark_equity(BENCHMARK_MODE, data, idx)
    equity_df["benchmark"] = bench_series
    bench_stats = compute_stats(equity_df["benchmark"])

    print("Backtest Stats (Strategy):")
    for k, v in stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")

    print("\nBenchmark (BASE) Stats:")
    for k, v in bench_stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")

    os.makedirs(OUT_DIR, exist_ok=True)
    equity_df.to_csv(f"{OUT_DIR}/equity_curve_ultra_crypto.csv")
    expo_df.to_csv(f"{OUT_DIR}/exposures_ultra_crypto.csv")
    trades_df.to_csv(f"{OUT_DIR}/trades_ultra_crypto.csv", index=False)

    compare_df = pd.DataFrame({"strategy": equity_df["equity"], "benchmark": equity_df["benchmark"]}, index=equity_df.index)
    compare_df.to_csv(f"{OUT_DIR}/compare_equity_ultra_crypto.csv")

    _write_run_manifest(stats, bench_stats, list(data.keys()))
    print(f"\nSaved: {OUT_DIR}/equity_curve_ultra_crypto.csv, {OUT_DIR}/compare_equity_ultra_crypto.csv, {OUT_DIR}/exposures_ultra_crypto.csv, {OUT_DIR}/trades_ultra_crypto.csv, {OUT_DIR}/run_crypto_ultra.json")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        equity_df[["equity", "benchmark"]].plot()
        plt.title("ULTRA Crypto Momentum: Strategy vs BTC-USD")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        fp = f"{OUT_DIR}/equity_curve_ultra_crypto.png"
        plt.savefig(fp)
        plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__ == "__main__":
    main()
