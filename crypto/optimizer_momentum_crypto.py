# AME-V4.6-ULTRA-CRYPTO Optimizer (CAGR-focused with Drawdown-Duration Control)
# Objective: MAX CAGR with constraint: avoid prolonged drawdowns.
# - Hard fitness penalty if drawdown duration > 30 days (any time under high-water mark).
# - Additional penalty if last-5Y CAGR < 200% (>= 2.0).
# - Ultra-aggressive momentum; daily rebal; cash fallback when momentum collapses.
# Python 3.10+
# Deps: pandas, numpy, yfinance, matplotlib (optional)

import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------- Data ------------------------------------ #

BASE_SYMBOL = "BTC-USD"
YF_PERIOD = "max"
START_CAPITAL = 100_000.0
OUT_DIR = "./out"

UNIVERSE = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "AVAX-USD", "MATIC-USD", "DOT-USD",
    "LTC-USD", "BCH-USD", "LINK-USD", "ATOM-USD", "UNI-USD",
    "XLM-USD", "ETC-USD",
    "USDT-USD", "USDC-USD",
]

SLIPPAGE_BPS = 10
FEE_BPS = 10
QTY_DECIMALS = 8

MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000

# Constraint targets
MAX_DD_DURATION_DAYS = 30
REQUIRED_CAGR_5Y = 2.0  # 200% annualized over last 5 years

# --------------------------- Utilities ----------------------------------- #

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
        raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo")
    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

    # Soft liquidity screen
    def liq_ok(df: pd.DataFrame) -> pd.Series:
        adv = (df["close"] * df["volume"]).rolling(30, min_periods=30).mean()
        daily = df["close"] * df["volume"]
        return (adv > MIN_DOLLAR_VOL_AVG_30D) & (daily > MIN_DAILY_DOLLAR_VOL)

    keep = {}
    for s, df in data.items():
        m = liq_ok(df)
        if m.sum() >= int(0.6 * len(m)):
            keep[s] = df
    if BASE_SYMBOL not in keep:
        keep[BASE_SYMBOL] = data[BASE_SYMBOL]
    return keep

def pct_change_series(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods=periods, fill_method=None)

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip)
    ref = past.shift(lookback)
    return past.divide(ref) - 1.0

def apply_cost(price: float, side: str) -> float:
    slip = price * (SLIPPAGE_BPS / 10_000)
    fee = price * (FEE_BPS / 10_000)
    return price + slip + fee if side == "buy" else price - slip - fee

def round_qty(q: float) -> float:
    if not np.isfinite(q) or q <= 0: return 0.0
    return float(np.floor(q * (10 ** QTY_DECIMALS)) / (10 ** QTY_DECIMALS))

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
        raise ValueError("freq must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)

# ------------------------ Strategy (param-driven) ------------------------ #

@dataclass(frozen=True)
class Params:
    lookbacks: Tuple[int, int, int]
    weights: Tuple[float, float, float]
    skip_recent: int
    rebalance: str
    allow_three_when_spread: bool
    spread_threshold1: float
    spread_threshold2: float
    abs_mom_gate_base: float
    max_weight_min: float
    max_weight_max: float  # leave wide; user doesn't care
    cash_buffer_min: float
    cash_buffer_max: float
    universe_mode: str

@dataclass
class Result:
    params: Params
    stats: Dict[str, float]
    bench: Dict[str, float]

def build_panel(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s: df["close"] for s, df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s: df["open"] for s, df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def momentum_score(close: pd.DataFrame, p: Params) -> pd.DataFrame:
    l1, l2, l3 = p.lookbacks
    w1, w2, w3 = p.weights
    m1 = trailing_return(close, l1, p.skip_recent)
    m2 = trailing_return(close, l2, p.skip_recent)
    m3 = trailing_return(close, l3, p.skip_recent)
    return w1 * m1 + w2 * m2 + w3 * m3

# ------------------------ Cash-Fallback Rules ---------------------------- #

def market_guard(dt: pd.Timestamp, close: pd.DataFrame, mom_row: pd.Series) -> bool:
    """
    Aggressive circuit breaker to avoid prolonged drawdowns:
    - If the best momentum is <= abs_gate*0.5 (≈ non-existent) AND
    - BTC 14d return < 0 AND cross-sectional median 14d return < 0
    => Go to cash this rebalance.
    """
    # 14d returns
    try:
        btc = close["BTC-USD"]
    except Exception:
        return False
    # guard when recent returns are decisively negative
    if dt not in btc.index: return False
    idx_pos = close.index.get_loc(dt)
    if idx_pos < 14: return False
    r14_btc = float(btc.iloc[idx_pos] / btc.iloc[idx_pos - 14] - 1.0)
    # breadth via median of top universe 14d ret
    r14_all = close.iloc[idx_pos] / close.iloc[idx_pos - 14] - 1.0
    med14 = float(np.nanmedian(r14_all.values))

    # best raw momentum signal
    top_mom = float(np.nanmax(mom_row.values))

    # thresholds
    weak_mom = (top_mom <= 0.0)  # super strict: require positive momentum to be long
    bad_tape = (r14_btc < 0.0) and (med14 < 0.0)

    return bool(weak_mom and bad_tape)

# ------------------------------ Backtest --------------------------------- #

def run_strategy(data: Dict[str, pd.DataFrame], p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx, close, openp = build_panel(data)
    mom = momentum_score(close, p)

    reb_days = last_day_per_period(idx, 'D')

    largest_lb = max(p.lookbacks) + p.skip_recent + 10
    first_allowed = idx[0] + pd.Timedelta(days=largest_lb)
    reb_days = reb_days[reb_days >= first_allowed]

    defensive_set = {"USDT-USD", "USDC-USD"}

    cash = START_CAPITAL
    positions: Dict[str, float] = {}
    equity = pd.Series(index=idx, dtype=float)

    def pv(dt: pd.Timestamp) -> float:
        v = cash
        if positions:
            syms = list(positions.keys())
            px = close.loc[dt, syms]
            v += float(np.nansum(px.values * np.array([positions[s] for s in syms], dtype=float)))
        return v

    mom_np = mom.to_numpy(copy=False)
    cols = list(mom.columns)
    col_is_def = np.array([c in defensive_set for c in cols], dtype=bool)

    for i, dt in enumerate(idx):
        equity.iloc[i] = pv(dt)
        if dt not in reb_days or i + 1 >= len(idx):
            continue

        t_exec = idx[i + 1]
        port_val = pv(dt)

        # Momentum row and guard
        row = mom_np[i, :]
        row_series = pd.Series(row, index=cols)
        if market_guard(dt, close, row_series):
            # Full cash this rebalance to cut drawdown duration
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty > 0:
                        exec_px = apply_cost(opx, "sell")
                        cash += qty * exec_px
                positions.clear()
            continue

        valid = np.isfinite(row) & (~col_is_def)
        gated = valid & (row > p.abs_mom_gate_base)

        if gated.any():
            order = np.argsort(row[gated])[::-1]
            elig_idx = np.where(gated)[0][order]
        elif valid.any():
            order_all = np.argsort(row[valid])[::-1]
            elig_idx = np.where(valid)[0][order_all]
        else:
            # liquidate to cash
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty > 0:
                        exec_px = apply_cost(opx, "sell")
                        cash += qty * exec_px
                positions.clear()
            continue

        elig_syms = [cols[j] for j in elig_idx]
        top_vals = [float(row[j]) if np.isfinite(row[j]) else -np.inf for j in elig_idx[:3]]

        picks: List[str] = [elig_syms[0]]
        spread = 0.0

        if len(elig_syms) >= 2:
            if (top_vals[0] - top_vals[1]) >= p.spread_threshold1:
                picks.append(elig_syms[1])
                spread = max(spread, top_vals[0] - top_vals[1])

        if p.allow_three_when_spread and len(elig_syms) >= 3:
            if (top_vals[1] - top_vals[2]) >= p.spread_threshold2:
                picks.append(elig_syms[2])
                spread = max(spread, top_vals[1] - top_vals[2])

        # dynamic sizing
        spread_clip = max(0.0, min(0.20, spread))
        frac = spread_clip / 0.20
        max_w = p.max_weight_min + (p.max_weight_max - p.max_weight_min) * frac
        cash_buf = p.cash_buffer_max - (p.cash_buffer_max - p.cash_buffer_min) * frac

        if len(picks) == 1:
            target_w = {picks[0]: min(max_w, 1.0 - cash_buf)}
        elif len(picks) == 2:
            raw = {picks[0]: 1.0, picks[1]: 0.9}
            ssum = sum(raw.values())
            raw = {k: v / ssum for k, v in raw.items()}
            capped = {k: min(max_w, v) for k, v in raw.items()}
            ssum = sum(capped.values())
            target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {
                picks[0]: 1.0 - cash_buf
            }
        else:
            raw = {picks[0]: 1.0, picks[1]: 0.9, picks[2]: 0.8}
            ssum = sum(raw.values())
            raw = {k: v / ssum for k, v in raw.items()}
            capped = {k: min(max_w, v) for k, v in raw.items()}
            ssum = sum(capped.values())
            target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {
                picks[0]: 1.0 - cash_buf
            }

        target_notional = {s: w * port_val for s, w in target_w.items()}
        cur_notional = {s: positions.get(s, 0.0) * float(close.loc[dt, s]) for s in positions.keys()}
        all_syms = sorted(set(list(cur_notional.keys()) + list(target_notional.keys())))

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
                    exec_px = apply_cost(opx, "sell")
                    cash += qty * exec_px
                    positions[s] = positions.get(s, 0.0) - qty
                    if positions[s] <= 0: positions.pop(s, None)

        # Buys
        for s in all_syms:
            opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt = target_notional.get(s, 0.0)
            cur = cur_notional.get(s, 0.0)
            diff = tgt - cur
            if diff > 1e-8 and cash > 0:
                exec_px = apply_cost(opx, "buy")
                qty = round_qty(min(diff, cash) / exec_px)
                if qty > 0:
                    cost = qty * exec_px
                    if cost <= cash:
                        cash -= cost
                        positions[s] = positions.get(s, 0.0) + qty

    if len(idx) > 0:
        equity.iloc[-1] = pv(idx[-1])
    equity_df = equity.to_frame("equity")

    # Benchmark: BTC buy & hold
    base_close = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    bench_equity = (START_CAPITAL * (base_close / base_close.iloc[0])).reindex(idx).ffill()
    equity_df["benchmark"] = bench_equity
    return equity_df, close

# ----------------------- Fitness & Statistics ---------------------------- #

def compute_stats(equity: pd.Series) -> Dict[str, float]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {k: np.nan for k in ["CAGR", "TotalReturn", "MaxDD", "Vol", "Sharpe", "MAR",
                                    "MaxDDDuration", "CAGR_5Y"]}
    rets = eq.pct_change().iloc[1:]
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total) ** (1 / years) - 1 if years and years > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0)
    maxdd = float(dd.min()) if len(dd) else np.nan
    vol = float(rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan
    mar = float(cagr / abs(maxdd)) if (isinstance(maxdd, float) and maxdd < 0 and not np.isnan(cagr)) else np.nan

    # Drawdown duration (longest stretch under previous peak)
    under = dd < 0
    # compute longest run length
    max_len = 0
    cur_len = 0
    for flag in under.astype(int).values:
        if flag:
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
        else:
            cur_len = 0

    # Last-5Y CAGR
    end_date = eq.index[-1]
    start_5y = end_date - pd.Timedelta(days=int(365.25 * 5))
    eq5 = eq[eq.index >= start_5y]
    if len(eq5) >= 2:
        total5 = float(eq5.iloc[-1] / eq5.iloc[0] - 1.0)
        yrs5 = (eq5.index[-1] - eq5.index[0]).days / 365.25
        cagr5 = (1 + total5) ** (1 / yrs5) - 1 if yrs5 > 0 else np.nan
    else:
        cagr5 = np.nan

    return {"CAGR": cagr, "TotalReturn": total, "MaxDD": maxdd, "Vol": vol,
            "Sharpe": sharpe, "MAR": mar, "MaxDDDuration": float(max_len), "CAGR_5Y": cagr5}

def fitness(res: Result) -> float:
    cagr = res.stats.get("CAGR", np.nan)
    dd = res.stats.get("MaxDD", np.nan)
    sharpe = res.stats.get("Sharpe", np.nan)
    bench_cagr = res.bench.get("CAGR", np.nan)
    dd_dur = res.stats.get("MaxDDDuration", np.nan)
    cagr5 = res.stats.get("CAGR_5Y", np.nan)

    if any(np.isnan(x) for x in [cagr, dd, bench_cagr, dd_dur, cagr5]):
        return -1e9

    # Base edge on CAGR vs BTC
    edge = cagr - bench_cagr

    # Penalties
    pen = 0.0

    # Hard penalty for prolonged drawdown duration
    if dd_dur > MAX_DD_DURATION_DAYS:
        pen += (dd_dur - MAX_DD_DURATION_DAYS) * 15.0  # strong per-day penalty

    # Soft penalty for very deep DD (still tolerate aggression)
    if dd < -0.85:
        pen += (abs(dd) - 0.85) * 50.0

    # Enforce last-5Y CAGR >= 200%
    if cagr5 < REQUIRED_CAGR_5Y:
        pen += (REQUIRED_CAGR_5Y - cagr5) * 500.0  # heavy shortfall penalty

    return edge * 260.0 + (sharpe if not np.isnan(sharpe) else 0.0) * 2.0 - pen

def evaluate(data: Dict[str, pd.DataFrame], p: Params) -> Result:
    eq, _ = run_strategy(data, p)
    stats = compute_stats(eq["equity"])
    bench = compute_stats(eq["benchmark"])
    return Result(params=p, stats=stats, bench=bench)

# --------------------------- Search Space -------------------------------- #

LB_OPTIONS = [
    (63, 21, 7), (42, 14, 7), (28, 14, 7),
    (21, 10, 5), (14, 7, 3),
    (10, 5, 2), (7, 3, 1), (5, 2, 1), (3, 1, 0),
]
WT_OPTIONS = [
    (0.9, 0.08, 0.02),
    (0.8, 0.15, 0.05),
    (0.7, 0.2, 0.1),
    (0.6, 0.3, 0.1),
    (0.5, 0.3, 0.2),
]
SKIP_OPTIONS = [0, 1, 2]
GATE_OPTIONS = [-0.02, 0.000, 0.005]  # slightly less negative than V4 to reduce whipsaw risk
SPREAD_OPTIONS = [0.001, 0.003, 0.005]

def random_params() -> Params:
    return Params(
        lookbacks=random.choice(LB_OPTIONS),
        weights=random.choice(WT_OPTIONS),
        skip_recent=random.choice(SKIP_OPTIONS),
        rebalance='D',
        allow_three_when_spread=True,
        spread_threshold1=random.choice(SPREAD_OPTIONS),
        spread_threshold2=random.choice(SPREAD_OPTIONS),
        abs_mom_gate_base=random.choice(GATE_OPTIONS),
        max_weight_min=1.00,
        max_weight_max=2.50,  # user doesn't care; keep wide
        cash_buffer_min=0.00,
        cash_buffer_max=0.05,  # allow small buffer to help duration
        universe_mode="EXT",
    )

def mutate(p: Params, rate: float = 0.55) -> Params:
    def flip(cur, opts):
        return random.choice([o for o in opts if o != cur]) if random.random() < rate else cur
    return Params(
        lookbacks=flip(p.lookbacks, LB_OPTIONS),
        weights=flip(p.weights, WT_OPTIONS),
        skip_recent=flip(p.skip_recent, SKIP_OPTIONS),
        rebalance='D',
        allow_three_when_spread=True,
        spread_threshold1=flip(p.spread_threshold1, SPREAD_OPTIONS),
        spread_threshold2=flip(p.spread_threshold2, SPREAD_OPTIONS),
        abs_mom_gate_base=flip(p.abs_mom_gate_base, GATE_OPTIONS),
        max_weight_min=1.00,
        max_weight_max=2.50,
        cash_buffer_min=0.00,
        cash_buffer_max=0.05,
        universe_mode=p.universe_mode,
    )

def crossover(a: Params, b: Params) -> Params:
    pick = lambda x, y: random.choice([x, y])
    return Params(
        lookbacks=pick(a.lookbacks, b.lookbacks),
        weights=pick(a.weights, b.weights),
        skip_recent=pick(a.skip_recent, b.skip_recent),
        rebalance='D',
        allow_three_when_spread=True,
        spread_threshold1=pick(a.spread_threshold1, b.spread_threshold1),
        spread_threshold2=pick(a.spread_threshold2, b.spread_threshold2),
        abs_mom_gate_base=pick(a.abs_mom_gate_base, b.abs_mom_gate_base),
        max_weight_min=1.00,
        max_weight_max=2.50,
        cash_buffer_min=0.00,
        cash_buffer_max=0.05,
        universe_mode=a.universe_mode,
    )

# --------------------------- Evolution Loop ------------------------------ #

def evolutionary_search(
        data: Dict[str, pd.DataFrame],
        population_size: int = 120,
        generations: int = 220,
        elite_frac: float = 0.10,
        mutation_rate: float = 0.55,
        seed: Optional[int] = 42,
) -> Tuple[List[Result], Result]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    pop = [random_params() for _ in range(population_size)]
    results: List[Result] = []

    for gen in range(generations):
        gen_results = [evaluate(data, p) for p in pop]
        gen_results.sort(key=lambda r: fitness(r), reverse=True)
        best = gen_results[0]

        def fmt_pct(x: float) -> str:
            return f"{x * 100:.2f}%" if np.isfinite(x) else "nan"

        print(
            f"[Gen {gen + 1}/{generations}] "
            f"Score={fitness(best):.2f} | "
            f"CAGR={fmt_pct(best.stats['CAGR'])} vs BTC {fmt_pct(best.bench['CAGR'])} | "
            f"DD={fmt_pct(best.stats['MaxDD'])} | "
            f"DD_Dur={best.stats['MaxDDDuration']:.0f}d | "
            f"CAGR_5Y={fmt_pct(best.stats['CAGR_5Y'])} | "
            f"LBs={best.params.lookbacks} | Wts={best.params.weights} | "
            f"AbsGate>{best.params.abs_mom_gate_base * 100:.2f}% | Reb={best.params.rebalance} | "
            f"MaxW=[{best.params.max_weight_min:.2f},{best.params.max_weight_max:.2f}] | "
            f"Spread>{best.params.spread_threshold1 * 100:.2f}%/{best.params.spread_threshold2 * 100:.2f}%"
        )
        results.extend(gen_results)

        n_elite = max(1, int(elite_frac * population_size))
        elite_params = [r.params for r in gen_results[:n_elite]]
        parent_pool = [r.params for r in gen_results[: max(n_elite * 4, n_elite + 8)]]

        children: List[Params] = []
        while len(children) < population_size - n_elite:
            pa = random.choice(elite_params)
            pb = random.choice(parent_pool)
            child = crossover(pa, pb)
            child = mutate(child, rate=mutation_rate)
            children.append(child)

        pop = elite_params + children

    results.sort(key=lambda r: fitness(r), reverse=True)
    return results, results[0]

# ------------------------------- I/O ------------------------------------- #

def save_results(all_results: List[Result], best: Result, tag: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for r in all_results:
        rows.append({
            "score": fitness(r),
            "CAGR": r.stats.get("CAGR"),
            "TotalReturn": r.stats.get("TotalReturn"),
            "MaxDD": r.stats.get("MaxDD"),
            "Sharpe": r.stats.get("Sharpe"),
            "MAR": r.stats.get("MAR"),
            "Bench_CAGR": r.bench.get("CAGR"),
            "MaxDDDuration": r.stats.get("MaxDDDuration"),
            "CAGR_5Y": r.stats.get("CAGR_5Y"),
            "lookbacks": r.params.lookbacks,
            "weights": r.params.weights,
            "skip_recent": r.params.skip_recent,
            "rebalance": r.params.rebalance,
            "allow_three_when_spread": r.params.allow_three_when_spread,
            "spread_threshold1": r.params.spread_threshold1,
            "spread_threshold2": r.params.spread_threshold2,
            "abs_mom_gate_base": r.params.abs_mom_gate_base,
            "max_weight_min": r.params.max_weight_min,
            "max_weight_max": r.params.max_weight_max,
            "cash_buffer_min": r.params.cash_buffer_min,
            "cash_buffer_max": r.params.cash_buffer_max,
            "universe_mode": r.params.universe_mode,
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(400)
    df.to_csv(f"{OUT_DIR}/optimizer_results_ame_v4p6_ultra_crypto.csv", index=False)

    manifest = {
        "best_params": {
            "lookbacks": best.params.lookbacks,
            "weights": best.params.weights,
            "skip_recent": best.params.skip_recent,
            "rebalance": best.params.rebalance,
            "allow_three_when_spread": best.params.allow_three_when_spread,
            "spread_threshold1": best.params.spread_threshold1,
            "spread_threshold2": best.params.spread_threshold2,
            "abs_mom_gate_base": best.params.abs_mom_gate_base,
            "max_weight_min": best.params.max_weight_min,
            "max_weight_max": best.params.max_weight_max,
            "cash_buffer_min": best.params.cash_buffer_min,
            "cash_buffer_max": best.params.cash_buffer_max,
            "universe_mode": best.params.universe_mode,
        },
        "best_stats": best.stats,
        "best_benchmark": best.bench,
        "constraints": {
            "MaxDDDurationDays": MAX_DD_DURATION_DAYS,
            "RequiredCAGR5Y": REQUIRED_CAGR_5Y,
        }
    }
    with open(f"{OUT_DIR}/optimizer_manifest_ame_v4p6_ultra_crypto.json", "w") as f:
        json.dump(manifest, f, indent=2)

# -------------------------------- Main ----------------------------------- #

def main():
    universe = _dedupe(UNIVERSE)
    print(f"Downloading {len(universe)} tickers...")
    data = fetch_data(universe)
    print(f"Universe after hygiene: {len(data)} tickers")

    all_results, best = evolutionary_search(
        data,
        population_size=120,
        generations=220,
        elite_frac=0.10,
        mutation_rate=0.55,
        seed=42,
    )

    save_results(all_results, best, tag="v4p6_ultra_crypto")

    print("\nBest Configuration:")
    print(best.params)
    print("Strategy Stats:")
    for k, v in best.stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")
    print("Benchmark (BTC-USD) Stats:")
    for k, v in best.bench.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}")
    print(f"Score: {fitness(best):.2f}")

    try:
        eq, _ = run_strategy(data, best.params)
        import matplotlib.pyplot as plt
        plt.figure()
        eq[["equity", "benchmark"]].plot()
        plt.title("AME-V4.6-ULTRA-CRYPTO Best Strategy vs BTC-USD")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        fp = f"{OUT_DIR}/optimizer_ame_v4p6_ultra_crypto_best_equity.png"
        plt.savefig(fp)
        plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__ == "__main__":
    main()
