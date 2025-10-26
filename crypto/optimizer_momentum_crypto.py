# AME-V3-HV-CRYPTO Optimizer
# Spot-only, no leverage/derivatives. Built to maximize CAGR with pragmatic DD tolerance.
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

# Liquid spot crypto universe (USD pairs, no leverage)
UNIVERSE = [
    # majors
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "AVAX-USD", "MATIC-USD", "DOT-USD",
    "LTC-USD", "BCH-USD", "LINK-USD", "ATOM-USD", "UNI-USD",
    "XLM-USD", "ETC-USD",
    # defensives / cash proxies
    "USDT-USD", "USDC-USD",
]

# Trading costs (spot, conservative)
SLIPPAGE_BPS = 10
FEE_BPS = 10
QTY_DECIMALS = 8

# Liquidity guards (soft)
MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000

# --------------------------- Utilities ----------------------------------- #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    symbols = _dedupe(symbols)
    raw = yf.download(
        symbols,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    data: Dict[str, pd.DataFrame] = {}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        d = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        ).copy()
        d = d.dropna(subset=["open", "high", "low", "close", "volume"])
        bad = (
                (d["open"] <= 0)
                | (d["high"] <= 0)
                | (d["low"] <= 0)
                | (d["close"] <= 0)
                | (d["volume"] < 0)
        )
        d = d[~bad]
        d.index.name = "date"
        return d[["open", "high", "low", "close", "volume"]]

    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        for s in symbols:
            if s not in top:
                continue
            data[s] = clean(raw[s].copy())
    else:
        s = symbols[0]
        data[s] = clean(raw)

    if BASE_SYMBOL not in data:
        raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo")
    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

    # soft liquidity screen
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
    if not np.isfinite(q) or q <= 0:
        return 0.0
    return float(np.floor(q * (10**QTY_DECIMALS)) / (10**QTY_DECIMALS))


def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f = freq.upper()
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
    # core momentum
    lookbacks: Tuple[int, int, int]  # e.g. (21,10,5)
    weights: Tuple[float, float, float]
    skip_recent: int  # 0/1/2
    rebalance: str  # 'D' or 'W'
    n_positions_base: int  # 1 (kept for logging)
    allow_two_when_spread: bool
    allow_three_when_spread: bool
    spread_threshold1: float
    spread_threshold2: float

    # gating & sizing
    abs_mom_gate_base: float
    max_weight_min: float
    max_weight_max: float
    cash_buffer_min: float
    cash_buffer_max: float

    # universe mode placeholder
    universe_mode: str  # 'EXT'


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


def dynamic_max_weight(p: Params, spread: float) -> float:
    spread = max(0.0, min(0.20, spread))
    frac = spread / 0.20
    return p.max_weight_min + (p.max_weight_max - p.max_weight_min) * frac


def dynamic_cash_buffer(p: Params, spread: float) -> float:
    spread = max(0.0, min(0.20, spread))
    frac = spread / 0.20
    return p.cash_buffer_max - (p.cash_buffer_max - p.cash_buffer_min) * frac


def run_strategy(data: Dict[str, pd.DataFrame], p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx, close, openp = build_panel(data)
    mom = momentum_score(close, p)

    # Rebalance cadence
    reb_days = last_day_per_period(idx, p.rebalance or 'D')

    largest_lb = max(p.lookbacks) + p.skip_recent + 10
    first_allowed = idx[0] + pd.Timedelta(days=largest_lb)
    reb_days = reb_days[reb_days >= first_allowed]

    defensive_set = {"USDT-USD", "USDC-USD"}

    cash = START_CAPITAL
    positions: Dict[str, float] = {}
    equity = pd.Series(index=idx, dtype=float)

    def pv(dt: pd.Timestamp) -> float:
        v = cash
        # fast vector valuation
        if positions:
            syms = list(positions.keys())
            px = close.loc[dt, syms]
            v += float(np.nansum(px.values * np.array([positions[s] for s in syms], dtype=float)))
        return v

    # Precompute eligible mask per day for speed
    mom_np = mom.to_numpy(copy=False)
    cols = list(mom.columns)
    col_is_def = np.array([c in defensive_set for c in cols], dtype=bool)

    for i, dt in enumerate(idx):
        equity.iloc[i] = pv(dt)
        if dt not in reb_days or i + 1 >= len(idx):
            continue

        t_exec = idx[i + 1]
        port_val = pv(dt)

        # Eligible set: apply absolute momentum gate; exclude defensives
        row = mom_np[i, :]
        valid = np.isfinite(row) & (~col_is_def)

        if valid.any():
            gated = valid & (row > p.abs_mom_gate_base)
        else:
            gated = valid

        # If nothing passes the gate, still force top-1 by raw momentum (hyper-growth mode)
        picks_syms: List[str] = []
        spread = 0.0

        if gated.any():
            # rank by momentum score
            order = np.argsort(row[gated])[::-1]
            elig_syms = [cols[j] for j in np.where(gated)[0][order]]
        else:
            # fallback to best raw momentum ignoring gate (can be negative)
            if valid.any():
                order_all = np.argsort(row[valid])[::-1]
                elig_syms = [cols[j] for j in np.where(valid)[0][order_all]]
            else:
                elig_syms = []

        if not elig_syms:
            # liquidate to cash (rare if valid empty)
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty > 0:
                        exec_px = apply_cost(opx, "sell")
                        cash += qty * exec_px
                positions.clear()
            continue

        # Build picks with spread checks over momentum differences
        picks_syms.append(elig_syms[0])
        # Need momentum values for top ranks
        top_vals = []
        for k in range(min(3, len(elig_syms))):
            s = elig_syms[k]
            j = cols.index(s)
            top_vals.append(float(row[j]) if np.isfinite(row[j]) else -np.inf)

        if p.allow_two_when_spread and len(elig_syms) >= 2:
            if (top_vals[0] - top_vals[1]) >= p.spread_threshold1:
                picks_syms.append(elig_syms[1])
                spread = max(spread, top_vals[0] - top_vals[1])

        if p.allow_three_when_spread and len(elig_syms) >= 3:
            if (top_vals[1] - top_vals[2]) >= p.spread_threshold2:
                picks_syms.append(elig_syms[2])
                spread = max(spread, top_vals[1] - top_vals[2])

        # dynamic sizing
        max_w = dynamic_max_weight(p, spread)
        cash_buf = dynamic_cash_buffer(p, spread)

        if len(picks_syms) == 1:
            target_w = {picks_syms[0]: min(max_w, 1.0 - cash_buf)}
        elif len(picks_syms) == 2:
            raw = {picks_syms[0]: 1.0, picks_syms[1]: 0.85}
            ssum = sum(raw.values())
            raw = {k: v / ssum for k, v in raw.items()}
            capped = {k: min(max_w, v) for k, v in raw.items()}
            ssum = sum(capped.values())
            target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {
                picks_syms[0]: 1.0 - cash_buf
            }
        else:
            raw = {picks_syms[0]: 1.0, picks_syms[1]: 0.85, picks_syms[2]: 0.70}
            ssum = sum(raw.values())
            raw = {k: v / ssum for k, v in raw.items()}
            capped = {k: min(max_w, v) for k, v in raw.items()}
            ssum = sum(capped.values())
            target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()} if ssum > 0 else {
                picks_syms[0]: 1.0 - cash_buf
            }

        # translate to notionals
        target_notional = {s: w * port_val for s, w in target_w.items()}
        cur_notional = {s: positions.get(s, 0.0) * float(close.loc[dt, s]) for s in positions.keys()}
        all_syms = sorted(set(list(cur_notional.keys()) + list(target_notional.keys())))

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
                    exec_px = apply_cost(opx, "sell")
                    cash += qty * exec_px
                    positions[s] = positions.get(s, 0.0) - qty
                    if positions[s] <= 0:
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

    # Benchmark: BTC buy&hold
    base_close = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    bench_equity = (START_CAPITAL * (base_close / base_close.iloc[0])).reindex(idx).ffill()
    equity_df["benchmark"] = bench_equity
    return equity_df, close

# ----------------------- Fitness & Statistics ---------------------------- #

def compute_stats(equity: pd.Series) -> Dict[str, float]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {k: np.nan for k in ["CAGR", "TotalReturn", "MaxDD", "Vol", "Sharpe", "MAR"]}
    rets = eq.pct_change().iloc[1:]
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (1 + total) ** (1 / years) - 1 if years and years > 0 else np.nan
    dd = (eq / eq.cummax() - 1.0)
    maxdd = float(dd.min()) if len(dd) else np.nan
    vol = float(rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365)) if rets.std() > 0 else np.nan
    mar = float(cagr / abs(maxdd)) if (
            isinstance(maxdd, float) and maxdd < 0 and cagr is not None and not np.isnan(cagr)
    ) else np.nan
    return {"CAGR": cagr, "TotalReturn": total, "MaxDD": maxdd, "Vol": vol, "Sharpe": sharpe, "MAR": mar}


def fitness(res: Result) -> float:
    cagr = res.stats.get("CAGR", np.nan)
    dd = res.stats.get("MaxDD", np.nan)
    sharpe = res.stats.get("Sharpe", np.nan)
    bench_cagr = res.bench.get("CAGR", np.nan)
    if any(np.isnan(x) for x in [cagr, dd, bench_cagr]):
        return -1e9
    edge = cagr - bench_cagr
    pen = 0.0
    # Allow deeper drawdowns before penalizing (hyper-growth mode)
    if dd < -0.85:
        pen += (abs(dd) - 0.85) * 8.0
    # Stronger weight on CAGR edge, some value to Sharpe
    return edge * 220.0 + (sharpe if not np.isnan(sharpe) else 0.0) * 5.0 - pen

def evaluate(data: Dict[str, pd.DataFrame], p: Params) -> Result:
    eq, _ = run_strategy(data, p)
    stats = compute_stats(eq["equity"])
    bench = compute_stats(eq["benchmark"])
    return Result(params=p, stats=stats, bench=bench)

# --------------------------- Search Space -------------------------------- #

LB_OPTIONS = [
    (14, 7, 3),
    (21, 10, 5),
    (28, 14, 7),
    (42, 14, 7),
    (63, 21, 7),
]
WT_OPTIONS = [
    (0.7, 0.2, 0.1),
    (0.6, 0.3, 0.1),
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2),
]
SKIP_OPTIONS = [0, 1, 2]
GATE_OPTIONS = [0.000, 0.005, 0.010]
SPREAD_OPTIONS = [0.003, 0.005, 0.010]
REBALANCE_OPTIONS = ['D', 'W']

def random_params() -> Params:
    return Params(
        lookbacks=random.choice(LB_OPTIONS),
        weights=random.choice(WT_OPTIONS),
        skip_recent=random.choice(SKIP_OPTIONS),
        rebalance=random.choice(REBALANCE_OPTIONS),
        n_positions_base=1,
        allow_two_when_spread=True,
        allow_three_when_spread=random.choice([True, False]),
        spread_threshold1=random.choice(SPREAD_OPTIONS),
        spread_threshold2=random.choice(SPREAD_OPTIONS),
        abs_mom_gate_base=random.choice(GATE_OPTIONS),
        max_weight_min=0.90,
        max_weight_max=1.50,
        cash_buffer_min=0.00,
        cash_buffer_max=0.05,
        universe_mode="EXT",
    )

def mutate(p: Params, rate: float = 0.35) -> Params:
    def flip(cur, opts):
        return random.choice([o for o in opts if o != cur]) if random.random() < rate else cur

    return Params(
        lookbacks=flip(p.lookbacks, LB_OPTIONS),
        weights=flip(p.weights, WT_OPTIONS),
        skip_recent=flip(p.skip_recent, SKIP_OPTIONS),
        rebalance=flip(p.rebalance, REBALANCE_OPTIONS),
        n_positions_base=1,
        allow_two_when_spread=True,
        allow_three_when_spread=flip(p.allow_three_when_spread, [True, False]),
        spread_threshold1=flip(p.spread_threshold1, SPREAD_OPTIONS),
        spread_threshold2=flip(p.spread_threshold2, SPREAD_OPTIONS),
        abs_mom_gate_base=flip(p.abs_mom_gate_base, GATE_OPTIONS),
        max_weight_min=0.90,
        max_weight_max=1.50,
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
        rebalance=pick(a.rebalance, b.rebalance),
        n_positions_base=1,
        allow_two_when_spread=True,
        allow_three_when_spread=pick(a.allow_three_when_spread, b.allow_three_when_spread),
        spread_threshold1=pick(a.spread_threshold1, b.spread_threshold1),
        spread_threshold2=pick(a.spread_threshold2, b.spread_threshold2),
        abs_mom_gate_base=pick(a.abs_mom_gate_base, b.abs_mom_gate_base),
        max_weight_min=0.90,
        max_weight_max=1.50,
        cash_buffer_min=0.00,
        cash_buffer_max=0.05,
        universe_mode=a.universe_mode,
    )

# --------------------------- Evolution Loop ------------------------------ #

def evolutionary_search(
        data: Dict[str, pd.DataFrame],
        population_size: int = 64,
        generations: int = 100,
        elite_frac: float = 0.25,
        mutation_rate: float = 0.35,
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
        print(
            f"[Gen {gen + 1}/{generations}] "
            f"Score={fitness(best):.2f} | "
            f"CAGR={best.stats['CAGR']:.2%} vs BTC {best.bench['CAGR']:.2%} | "
            f"DD={best.stats['MaxDD']:.2%} | N={best.params.n_positions_base} "
            f"| LBs={best.params.lookbacks} | Wts={best.params.weights} "
            f"| AbsGate>{best.params.abs_mom_gate_base:.2%} | Reb={best.params.rebalance} "
            f"| MaxW=[{best.params.max_weight_min:.2f},{best.params.max_weight_max:.2f}] "
            f"| Spread>{best.params.spread_threshold1:.2%}/{best.params.spread_threshold2:.2%}"
        )
        results.extend(gen_results)

        n_elite = max(1, int(elite_frac * population_size))
        elite_params = [r.params for r in gen_results[:n_elite]]

        children: List[Params] = []
        parent_pool = [r.params for r in gen_results[: max(n_elite * 3, n_elite + 5)]]
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
            "lookbacks": r.params.lookbacks,
            "weights": r.params.weights,
            "skip_recent": r.params.skip_recent,
            "rebalance": r.params.rebalance,
            "n_positions_base": r.params.n_positions_base,
            "allow_two_when_spread": r.params.allow_two_when_spread,
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
    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(300)
    df.to_csv(f"{OUT_DIR}/optimizer_results_ame_v3_hv_crypto.csv", index=False)

    manifest = {
        "best_params": {
            "lookbacks": best.params.lookbacks,
            "weights": best.params.weights,
            "skip_recent": best.params.skip_recent,
            "rebalance": best.params.rebalance,
            "n_positions_base": best.params.n_positions_base,
            "allow_two_when_spread": best.params.allow_two_when_spread,
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
    }
    with open(f"{OUT_DIR}/optimizer_manifest_ame_v3_hv_crypto.json", "w") as f:
        json.dump(manifest, f, indent=2)

# -------------------------------- Main ----------------------------------- #

def main():
    universe = _dedupe(UNIVERSE)
    print(f"Downloading {len(universe)} tickers...")
    data = fetch_data(universe)
    print(f"Universe after hygiene: {len(data)} tickers")

    all_results, best = evolutionary_search(
        data,
        population_size=64,
        generations=100,
        elite_frac=0.25,
        mutation_rate=0.35,
        seed=42,
    )

    save_results(all_results, best, tag="v3_hv_crypto")

    print("\nBest Configuration:")
    print(best.params)
    print("Strategy Stats:")
    for k, v in best.stats.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"{k}: {v:.4%}")
        else:
            print(f"{k}: {v}")
    print("Benchmark (BTC-USD) Stats:")
    for k, v in best.bench.items():
        if isinstance(v, float) and not np.isnan(v):
            print(f"{k}: {v:.4%}")
        else:
            print(f"{k}: {v}")
    print(f"Score: {fitness(best):.2f}")

    try:
        eq, _ = run_strategy(data, best.params)
        import matplotlib.pyplot as plt
        plt.figure()
        eq[["equity", "benchmark"]].plot()
        plt.title("AME-V3-HV-CRYPTO Best Strategy vs BTC-USD")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        fp = f"{OUT_DIR}/optimizer_ame_v3_hv_crypto_best_equity.png"
        plt.savefig(fp)
        plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__ == "__main__":
    main()
