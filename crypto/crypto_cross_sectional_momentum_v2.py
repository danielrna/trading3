# file: crypto_cross_sectional_momentum_v2.py
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
    "BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","TON-USD","LTC-USD","DOT-USD","TRX-USD","MATIC-USD"
]
UNIVERSE_EXT: List[str] = [
    "NEAR-USD","APT-USD","ARB-USD","OP-USD","ATOM-USD","ETC-USD","UNI-USD","SUI-USD","XLM-USD","AAVE-USD","INJ-USD","RUNE-USD","FIL-USD","HNT-USD","FTM-USD","TIA-USD"
]

BASE_SYMBOL = "BTC-USD"              # benchmark symbol and calendar anchor
YF_PERIOD = "max"
START_CAPITAL = 1000.0
OUT_DIR = "./out"

# ----------------- AME_V3 (light) — Best Params (frozen) ----------------- #
UNIVERSE_MODE = "EXT"                # "BASE" or "EXT"
LOOKBACKS: Tuple[int,int,int] = (42, 14, 7)
LOOKBACK_WEIGHTS: Tuple[float,float,float] = (0.4, 0.4, 0.2)
SKIP_RECENT_DAYS = 0
REBALANCE_FREQ = "D"                 # "D" | "W" | "2W" | "M"

N_POSITIONS_BASE = 1
ALLOW_TWO_WHEN_SPREAD = True
ALLOW_THREE_WHEN_SPREAD = True
SPREAD_THRESHOLD1 = 0.01             # enable #2 if top1 - top2 >= 1.0%
SPREAD_THRESHOLD2 = 0.0075           # enable #3 if top2 - top3 >= 0.75%

ABS_MOM_GATE_BASE = 0.003            # 0.30% absolute gate
ABS_MOM_GATE_SCALE_VOL = False       # scaling disabled (frozen)

MAX_WEIGHT_MIN = 0.80                # dynamic cap floor (via spread)
MAX_WEIGHT_MAX = 1.00                # dynamic cap ceiling (no extra boost)

CASH_BUFFER_MIN = 0.02               # 2% cash when spread large (confident)
CASH_BUFFER_MAX = 0.10               # 10% cash when spread tiny (cautious)

USE_REGIME = False                   # regime disabled (frozen)
SLOW_DOWN_IN_CHOP = False            # NA when regime disabled

DEFENSIVE_OVERRIDE = True            # fallback when nothing passes gate
DEFENSIVE_ASSET = "USDT-USD"         # stablecoin proxy

# Costs (per-side) — cryptos generally higher than ETFs
SLIPPAGE_BPS = 5
FEE_BPS = 2
QTY_DECIMALS = 6

# Liquidity guards (for crypto; Yahoo volume * close ~ dollar volume)
MIN_DOLLAR_VOL_AVG_30D = 50_000_000
MIN_DAILY_DOLLAR_VOL = 10_000_000

# Benchmark: "BTC" | "EQUAL"
BENCHMARK_MODE = "BTC"

# ============================== Data Layer =============================== #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    symbols = _dedupe(symbols + ["USDT-USD","USDC-USD"])  # ensure defensive presence if available
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
                (out["open"] <= 0) | (out["high"] <= 0) | (out["low"] <= 0) |
                (out["close"] <= 0) | (out["volume"] < 0)
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
    s = idx.to_series()
    if f == "M":
        g = s.groupby(idx.to_period("M")).max()
    elif f == "W":
        g = s.groupby(idx.to_period("W")).max()
    elif f == "2W":
        g = s.groupby(idx.to_period("W")).max().iloc[::2]
    elif f == "D":
        g = s
    else:
        raise ValueError("REBALANCE_FREQ must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)

def dynamic_caps(spread_top12: float) -> Tuple[float, float]:
    """
    Map spread in [0,10%] to:
      - weight cap in [MAX_WEIGHT_MIN, MAX_WEIGHT_MAX]
      - cash buffer in [CASH_BUFFER_MAX -> CASH_BUFFER_MIN] (inverse relation)
    """
    sp = max(0.0, min(0.10, spread_top12))
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
    Returns target weights and cash buffer based on:
      - Absolute momentum gate
      - Spread-based expansion to up to 3 positions
      - Dynamic caps and dynamic cash buffer by spread
    """
    scores = mom.loc[dt].dropna().copy()

    # Exclude defensive assets (stablecoins) from selection universe
    defensives = {"USDT-USD", "USDC-USD", "BUSD-USD", "DAI-USD"}
    scores = scores[[s for s in scores.index if s not in defensives and s in close.columns]]

    # Absolute gate
    gate = ABS_MOM_GATE_BASE if not ABS_MOM_GATE_SCALE_VOL else ABS_MOM_GATE_BASE  # scaling disabled
    eligible = [(s, float(sc)) for s, sc in scores.items() if np.isfinite(sc) and sc > gate]
    if not eligible:
        return {}, CASH_BUFFER_MAX

    eligible.sort(key=lambda x: x[1], reverse=True)
    chosen: List[str] = [eligible[0][0]]

    # Spread tiers
    spread1 = eligible[0][1] - (eligible[1][1] if len(eligible) >= 2 else -1e9)
    spread2 = (eligible[1][1] - eligible[2][1]) if len(eligible) >= 3 else -1e9
    if ALLOW_TWO_WHEN_SPREAD and len(eligible) >= 2 and spread1 >= SPREAD_THRESHOLD1:
        chosen.append(eligible[1][0])
    if ALLOW_THREE_WHEN_SPREAD and len(eligible) >= 3 and spread2 >= SPREAD_THRESHOLD2:
        chosen.append(eligible[2][0])

    # Dynamic cap and cash buffer from spread between top1 and top2
    cap, cash_buf = dynamic_caps(spread1 if np.isfinite(spread1) else 0.0)

    # Base raw weights favoring rank
    if len(chosen) == 1:
        target_w = {chosen[0]: min(cap, 1.0 - cash_buf)}
    else:
        raw = {chosen[0]: 1.0}
        if len(chosen) >= 2:
            raw[chosen[1]] = 0.85
        if len(chosen) >= 3:
            raw[chosen[2]] = 0.70
        ssum = sum(raw.values())
        raw = {k: v / ssum for k, v in raw.items()}
        capped = {k: min(cap, v) for k, v in raw.items()}
        ssum = sum(capped.values())
        target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {chosen[0]: 1.0 - cash_buf}

    return target_w, cash_buf

def run_backtest(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Universe pick
    syms = _dedupe(UNIVERSE_BASE if UNIVERSE_MODE.upper() == "BASE" else (UNIVERSE_BASE + UNIVERSE_EXT))
    syms = [s for s in syms if s in data]
    if BASE_SYMBOL not in syms:
        syms = [BASE_SYMBOL] + syms
    data_u = {s: data[s] for s in syms if s in data}

    idx, close, openp = build_panel(data_u)
    mom = composite_momentum(close, LOOKBACKS, LOOKBACK_WEIGHTS, SKIP_RECENT_DAYS)

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
                val += pos.qty * float(close.loc[t, s])
        return val

    for i, dt in enumerate(idx):
        equity.iloc[i] = portfolio_value(dt)

        if dt in reb_days and i + 1 < len(idx):
            t_exec = idx[i + 1]
            port_val = portfolio_value(dt)

            target_w, cash_buf = compute_target_weights(dt, close, mom)

            if not target_w:
                if DEFENSIVE_OVERRIDE and DEFENSIVE_ASSET in close.columns:
                    # Flush everything into defensive (respect cash buffer max)
                    investable = port_val * (1.0 - CASH_BUFFER_MAX)
                    opx = float(openp.loc[t_exec, DEFENSIVE_ASSET]) if DEFENSIVE_ASSET in openp.columns else np.nan
                    if np.isfinite(opx):
                        # Sell all current
                        for s, p in list(positions.items()):
                            opxs = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                            if np.isfinite(opxs) and p.qty > 0:
                                px = apply_trade_cost(opxs, "sell")
                                cash += p.qty * px
                                trades.append({"date": t_exec, "symbol": s, "side": "SELL", "qty": p.qty, "price": px, "reason": "defensive_override"})
                                positions.pop(s, None)
                        qty = round_qty(investable / apply_trade_cost(opx, "buy"))
                        if qty > 0:
                            cost = qty * apply_trade_cost(opx, "buy")
                            if cost <= cash:
                                cash -= cost
                                positions[DEFENSIVE_ASSET] = Position(symbol=DEFENSIVE_ASSET, qty=qty, entry_px=apply_trade_cost(opx, "buy"))
                                trades.append({"date": t_exec, "symbol": DEFENSIVE_ASSET, "side": "BUY", "qty": qty, "price": positions[DEFENSIVE_ASSET].entry_px, "reason": "defensive_override"})
                else:
                    # Go full cash
                    for s, p in list(positions.items()):
                        opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                        if np.isfinite(opx) and p.qty > 0:
                            px = apply_trade_cost(opx, "sell")
                            cash += p.qty * px
                            trades.append({"date": t_exec, "symbol": s, "side": "SELL", "qty": p.qty, "price": px, "reason": "rebalance_clear"})
                    positions.clear()
            else:
                # Build target notionals (respect dynamic cash buffer)
                target_notional = {s: float(w) * port_val for s, w in target_w.items()}
                cur_notional = {s: (pos.qty * float(close.loc[dt, s])) for s, pos in positions.items()}
                all_syms = sorted(set(cur_notional.keys()).union(target_notional.keys()))

                # Sells first
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
        close: Optional[pd.DataFrame] = None
) -> pd.Series:
    m = mode.upper().strip()
    if m == "BTC":
        sym = "BTC-USD"
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
        "LOOKBACKS": LOOKBACKS,
        "LOOKBACK_WEIGHTS": LOOKBACK_WEIGHTS,
        "SKIP_RECENT_DAYS": SKIP_RECENT_DAYS,
        "REBALANCE_FREQ": REBALANCE_FREQ,
        "N_POSITIONS_BASE": N_POSITIONS_BASE,
        "ALLOW_TWO_WHEN_SPREAD": ALLOW_TWO_WHEN_SPREAD,
        "ALLOW_THREE_WHEN_SPREAD": ALLOW_THREE_WHEN_SPREAD,
        "SPREAD_THRESHOLD1": SPREAD_THRESHOLD1,
        "SPREAD_THRESHOLD2": SPREAD_THRESHOLD2,
        "ABS_MOM_GATE_BASE": ABS_MOM_GATE_BASE,
        "ABS_MOM_GATE_SCALE_VOL": ABS_MOM_GATE_SCALE_VOL,
        "MAX_WEIGHT_MIN": MAX_WEIGHT_MIN,
        "MAX_WEIGHT_MAX": MAX_WEIGHT_MAX,
        "CASH_BUFFER_MIN": CASH_BUFFER_MIN,
        "CASH_BUFFER_MAX": CASH_BUFFER_MAX,
        "DEFENSIVE_OVERRIDE": DEFENSIVE_OVERRIDE,
        "DEFENSIVE_ASSET": DEFENSIVE_ASSET,
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

    # Benchmark
    bench_series = compute_benchmark_equity(BENCHMARK_MODE, data, idx, close_panel)
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