# file: optimizer_momentum_AME_V1.py
# Python 3.10+
# Deps: pandas, numpy, yfinance, matplotlib (optional)
# Purpose: AME-V1 (Adaptive Momentum Engine) optimizer to push CAGR while containing drawdowns.
# - No derivatives, no leveraged ETFs. Plain ETFs only.
# - Dynamic regime gating, adaptive abs momentum threshold, optional defensive override.
# - Dynamic position count (1→2) when dispersion is strong. Dynamic cash buffer & max weight.

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

BASE_SYMBOL = "VOO"
YF_PERIOD = "max"
START_CAPITAL = 100_000.0
OUT_DIR = "./out"

# Liquid, unlevered ETF universe (broad + sectors + tech + factors + themes + defensives)
UNIVERSE = [
    # broad
    "VOO", "VV", "VTI", "QQQ", "DIA", "IWM",
    # sectors (SPDRs)
    "XLK", "XLC", "XLY", "XLI", "XLF", "XLV", "XLP", "XLU", "XLB", "XLE", "XLRE",
    # tech & innovation
    "SMH", "SOXX", "IGV", "FDN",
    # factors / styles
    "MTUM", "QUAL", "VLUE", "SIZE",
    # thematics (still unlevered)
    "XBI", "IBB", "ICLN", "TAN", "PBW", "ARKK",
    # defensives / ballast
    "IEF", "TLT", "GLD", "IAU", "SHY"
]

# Volatility/regime helpers (non-derivative proxies)
# We'll try to fetch ^VIX; if unavailable, fall back to ATR-based proxy on VOO.
VIX_TICKER = "^VIX"

# Trading costs
SLIPPAGE_BPS = 1
FEE_BPS = 1
QTY_DECIMALS = 4

# Liquidity guards (ETFs are generally fine; keep soft guards)
MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000


# --------------------------- Utilities ----------------------------------- #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    symbols = _dedupe(symbols)
    raw = yf.download(symbols, period=period, interval="1d",
                      group_by="ticker", auto_adjust=False, threads=True, progress=False)
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
        for s in symbols:
            if s not in raw.columns.get_level_values(0):
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

    # liquidity mask (soft)
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
    # Future-proof pct_change deprecation defaults
    return s.pct_change(periods=periods, fill_method=None)


def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip)
    ref = past.shift(lookback)
    return past.divide(ref) - 1.0


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    r = pct_change_series(close, 1)
    return r.rolling(window, min_periods=window).std() * np.sqrt(252)


def apply_cost(price: float, side: str) -> float:
    slip = price * (SLIPPAGE_BPS / 10_000)
    fee = price * (FEE_BPS / 10_000)
    return price + slip + fee if side == "buy" else price - slip - fee


def round_qty(q: float) -> float:
    if not np.isfinite(q) or q <= 0: return 0.0
    return float(np.floor(q * (10 ** QTY_DECIMALS)) / (10 ** QTY_DECIMALS))


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
        g = s  # every day available
    else:
        raise ValueError("freq must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)


# -------------------------- Regime Detection ----------------------------- #

def try_fetch_vix(idx: pd.DatetimeIndex) -> Optional[pd.Series]:
    try:
        vix = yf.download(VIX_TICKER, period=YF_PERIOD, interval="1d", progress=False, auto_adjust=False)
        if vix is None or vix.empty:
            return None
        c = vix.get("Close")
        if c is None or c.empty:
            return None
        c.index.name = "date"
        return c.reindex(idx).ffill()
    except Exception:
        return None


def atr_vol_proxy(voo: pd.DataFrame, window: int = 14) -> pd.Series:
    # ATR% as a volatility proxy when VIX not available
    high, low, close = voo["high"], voo["low"], voo["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    atr_pct = atr / close
    return (atr_pct * 100.0).reindex(voo.index)


def compute_regime_series(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = data[BASE_SYMBOL].index
    out = pd.DataFrame(index=idx)

    # ---- Volatility source ----
    vix = try_fetch_vix(idx)
    if vix is None:
        vol = atr_vol_proxy(data[BASE_SYMBOL]) * 100  # fallback scaled
    else:
        vol = vix

    vol = vol.reindex(idx).ffill().squeeze()  # enforce Series
    out["vol_level"] = vol

    # ---- Smooth and slope ----
    vol_s = vol.rolling(10, min_periods=5).mean().squeeze()
    vol_slope = vol_s.pct_change(5).fillna(0.0).squeeze()

    # ---- Price distance from MA200 ----
    close = data[BASE_SYMBOL]["close"].reindex(idx).squeeze()
    ma200 = close.rolling(200, min_periods=200).mean().squeeze()
    dist = ((close / ma200) - 1.0).fillna(0.0).squeeze()

    # ---- Boolean regime logic ----
    risk_on = ((dist > 0) & (vol_slope <= 0)).astype(int)
    risk_off = ((dist < 0) & (vol_slope > 0)).astype(int)
    chop = (~((risk_on.astype(bool)) | (risk_off.astype(bool)))).astype(int)

    out["risk_on"] = risk_on
    out["risk_off"] = risk_off
    out["chop"] = chop

    return out


# ------------------------ Strategy (param-driven) ------------------------ #

@dataclass(frozen=True)
class Params:
    # core momentum
    lookbacks: Tuple[int, int, int]  # e.g. (84,21,7)
    weights: Tuple[float, float, float]  # sum ~ 1.0
    skip_recent: int  # e.g. 0 or 21
    rebalance: str  # 'D','W','2W','M'
    n_positions_base: int  # base number of positions (1)
    allow_two_when_spread: bool  # enable second position when dispersion high
    spread_threshold: float  # min spread between ranks to allow #2 (e.g. 0.02 = 2%)

    # gating & sizing
    abs_mom_gate_base: float  # base abs momentum gate (e.g. 0.01)
    abs_mom_gate_scale_vol: bool  # scale gate with vol regime
    max_weight_min: float  # lower bound of dynamic max weight (e.g. 0.8)
    max_weight_max: float  # upper bound of dynamic max weight (e.g. 1.0)
    cash_buffer_min: float  # min cash (e.g. 0.05)
    cash_buffer_max: float  # max cash (e.g. 0.20)

    # regime handling
    use_regime: bool  # enable regime logic
    slow_down_in_chop: bool  # switch to 2W rebalance in chop
    defensive_override: bool  # override to defensive basket in risk_off
    defensive_asset: str  # 'TLT','IEF','GLD','IAU','SHY'

    # universe
    universe_mode: str  # 'BASE' or 'EXT' (we always use UNIVERSE here, but left for extensibility)


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
    # (84,21,7) default style
    m1 = trailing_return(close, l1, p.skip_recent)
    m2 = trailing_return(close, l2, p.skip_recent)
    m3 = trailing_return(close, l3, p.skip_recent)
    return w1 * m1 + w2 * m2 + w3 * m3


def compute_abs_gate(gate_base: float, vol_level: float, scale: bool) -> float:
    if not scale or not np.isfinite(vol_level):
        return gate_base
    # Example scaling: when vol>20, gate rises linearly up to +1.5x by vol=35
    lo, hi = 20.0, 35.0
    mult = 1.0
    if vol_level <= lo:
        mult = 1.0
    elif vol_level >= hi:
        mult = 1.5
    else:
        mult = 1.0 + 0.5 * (vol_level - lo) / (hi - lo)
    return gate_base * mult


def dynamic_max_weight(p: Params, spread: float) -> float:
    # scale max weight based on dispersion (higher spread => allow higher weight)
    spread = max(0.0, min(0.10, spread))  # clamp 0..10%
    frac = spread / 0.10  # 0..1
    return p.max_weight_min + (p.max_weight_max - p.max_weight_min) * frac


def dynamic_cash_buffer(p: Params, spread: float) -> float:
    # tighter cash when conviction high
    spread = max(0.0, min(0.10, spread))
    frac = spread / 0.10
    # invert: more spread -> lower cash
    return p.cash_buffer_max - (p.cash_buffer_max - p.cash_buffer_min) * frac


def run_strategy(data: Dict[str, pd.DataFrame], p: Params) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx, close, openp = build_panel(data)
    mom = momentum_score(close, p)

    regime = compute_regime_series(data) if p.use_regime else None

    # baseline rebalance schedule (with optional slowdown in chop)
    def effective_reb_days():
        if not p.use_regime or not p.slow_down_in_chop:
            return last_day_per_period(idx, p.rebalance)
        # build daily set, then filter per regime state
        daily = last_day_per_period(idx, "D")
        if regime is None:
            return daily
        # if chop -> keep only every 10th trading day (≈ 2 weeks)
        mask = np.ones(len(daily), dtype=bool)
        chop_days = regime.loc[daily, "chop"].fillna(0).astype(int).values
        counter = 0
        for i in range(len(daily)):
            if chop_days[i] == 1:
                # allow only every 10th day in chop
                mask[i] = (counter % 10 == 0)
                counter += 1
            else:
                mask[i] = True
        return pd.DatetimeIndex(daily[mask])

    reb_days = effective_reb_days()

    # ensure sufficient history
    largest_lb = max(p.lookbacks) + p.skip_recent + 10
    first_allowed = idx[0] + pd.Timedelta(days=largest_lb)
    reb_days = reb_days[reb_days >= first_allowed]

    cash = START_CAPITAL
    positions: Dict[str, float] = {}
    equity = pd.Series(index=idx, dtype=float)

    def pv(dt: pd.Timestamp) -> float:
        v = cash
        for s, qty in positions.items():
            if s in close.columns:
                px = float(close.loc[dt, s])
                v += qty * px
        return v

    defensive_set = {"TLT", "IEF", "GLD", "IAU", "SHY"}

    for i, dt in enumerate(idx):
        equity.iloc[i] = pv(dt)
        if dt not in reb_days or i + 1 >= len(idx):
            continue

        t_exec = idx[i + 1]
        port_val = pv(dt)

        # Dynamic absolute momentum gate based on current vol
        vol_level = float(regime.loc[dt, "vol_level"]) if regime is not None else np.nan
        gate = compute_abs_gate(p.abs_mom_gate_base, vol_level, p.abs_mom_gate_scale_vol)

        # If risk_off and defensive_override -> allocate to defensive asset only
        if p.use_regime and p.defensive_override and regime.loc[dt, "risk_off"] == 1:
            if p.defensive_asset not in close.columns:
                # fallback to SHY if chosen asset missing
                asset = "SHY" if "SHY" in close.columns else BASE_SYMBOL
            else:
                asset = p.defensive_asset

            opx = float(openp.loc[t_exec, asset]) if asset in openp.columns else np.nan
            if np.isfinite(opx):
                # go 100% - keep min cash buffer
                target_cash = START_CAPITAL * 0.0  # we’ll handle via buffer below
                cash_buffer = p.cash_buffer_max  # be conservative in risk_off
                investable = port_val * (1.0 - cash_buffer)
                qty = round_qty(investable / opx)
                # liquidate others first
                for s, qty_old in list(positions.items()):
                    if s == asset: continue
                    opx_s = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx_s) and qty_old > 0:
                        exec_px = apply_cost(opx_s, "sell")
                        cash += qty_old * exec_px
                        positions.pop(s, None)
                if qty > 0:
                    # adjust cash vs current position on defensive asset
                    existing = positions.get(asset, 0.0)
                    add_qty = max(0.0, qty - existing)
                    cost = add_qty * apply_cost(opx, "buy")
                    if cost <= cash:
                        cash -= cost
                        positions[asset] = existing + add_qty
                continue

        # Build eligible set (exclude defensives for momentum selection)
        elig = []
        for s in close.columns:
            if s in defensive_set:
                continue
            sc = mom.loc[dt, s] if s in mom.columns else np.nan
            if not np.isfinite(sc) or sc <= gate:
                continue
            elig.append((s, float(sc)))

        if not elig:
            # nothing passes -> increase cash buffer and sell all
            for s, qty in list(positions.items()):
                opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                if np.isfinite(opx) and qty > 0:
                    exec_px = apply_cost(opx, "sell")
                    cash += qty * exec_px
            positions.clear()
            continue

        elig.sort(key=lambda x: x[1], reverse=True)
        best1 = elig[0]
        spread = 0.0
        choose = [best1[0]]
        if p.allow_two_when_spread and len(elig) >= 2:
            best2 = elig[1]
            # spread measured on score difference
            spread = max(0.0, best1[1] - best2[1])
            if spread >= p.spread_threshold:
                choose.append(best2[0])

        # dynamic sizing
        max_w = dynamic_max_weight(p, spread)
        cash_buf = dynamic_cash_buffer(p, spread)

        # target weights: if 1 name -> min(max_w, 1-cash_buf)
        # if 2 names -> split (capped) and renormalize to 1-cash_buf
        if len(choose) == 1:
            target_w = {choose[0]: min(max_w, 1.0 - cash_buf)}
        else:
            raw = {choose[0]: 1.0, choose[1]: 0.8}  # slight preference to #1
            ssum = sum(raw.values())
            raw = {k: v / ssum for k, v in raw.items()}
            # cap by max_w, then scale to (1 - cash_buf)
            capped = {k: min(max_w, v) for k, v in raw.items()}
            ssum = sum(capped.values())
            if ssum <= 0:
                target_w = {choose[0]: 1.0 - cash_buf}
            else:
                target_w = {k: (v / ssum) * (1.0 - cash_buf) for k, v in capped.items()}

        # translate to target dollar notionals
        target_notional = {s: target_w[s] * port_val for s in target_w}

        # current notionals
        cur_notional = {s: positions.get(s, 0.0) * float(close.loc[dt, s]) for s in positions.keys()}
        all_syms = sorted(set(list(cur_notional.keys()) + list(target_notional.keys())))

        # first sell excess
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

        # then buy shortfalls
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

    # Benchmark: VOO B&H aligned
    voo_close = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    voo_equity = (START_CAPITAL * (voo_close / voo_close.iloc[0])).reindex(idx).ffill()
    equity_df["benchmark"] = voo_equity
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
    vol = float(rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    mar = float(cagr / abs(maxdd)) if (
            isinstance(maxdd, float) and maxdd < 0 and cagr is not None and not np.isnan(cagr)) else np.nan
    return {"CAGR": cagr, "TotalReturn": total, "MaxDD": maxdd, "Vol": vol, "Sharpe": sharpe, "MAR": mar}


def fitness(res: Result) -> float:
    # Multi-objective: prioritize CAGR, penalize DD; reward MAR
    cagr = res.stats.get("CAGR", np.nan)
    dd = res.stats.get("MaxDD", np.nan)
    mar = res.stats.get("MAR", np.nan)
    bench_cagr = res.bench.get("CAGR", np.nan)
    if any(np.isnan(x) for x in [cagr, dd, bench_cagr]):
        return -1e9
    edge = cagr - bench_cagr
    # penalties
    pen = 0.0
    if dd < -0.35:
        pen += (abs(dd) - 0.35) * 4.0  # increasing penalty past -35%
    return edge * 100.0 + (mar if not np.isnan(mar) else 0.0) * 5.0 - pen


def evaluate(data: Dict[str, pd.DataFrame], p: Params) -> Result:
    eq, _ = run_strategy(data, p)
    stats = compute_stats(eq["equity"])
    bench = compute_stats(eq["benchmark"])
    return Result(params=p, stats=stats, bench=bench)


# --------------------------- Search Space -------------------------------- #

def random_params() -> Params:
    return Params(
        lookbacks=random.choice([(84, 21, 7), (63, 21, 7), (126, 42, 21)]),
        weights=random.choice([(0.4, 0.4, 0.2), (0.5, 0.4, 0.1), (0.6, 0.3, 0.1)]),
        skip_recent=random.choice([0, 21]),
        rebalance=random.choice(["D", "W", "2W", "M"]),
        n_positions_base=1,
        allow_two_when_spread=True,
        spread_threshold=random.choice([0.01, 0.015, 0.02, 0.03]),
        abs_mom_gate_base=random.choice([0.005, 0.01, 0.015]),
        abs_mom_gate_scale_vol=True,
        max_weight_min=random.choice([0.7, 0.8]),
        max_weight_max=random.choice([0.9, 1.0]),
        cash_buffer_min=random.choice([0.05, 0.08]),
        cash_buffer_max=random.choice([0.15, 0.20]),
        use_regime=True,
        slow_down_in_chop=True,
        defensive_override=True,
        defensive_asset=random.choice(["TLT", "IEF", "GLD", "IAU", "SHY"]),
        universe_mode="EXT",
    )


def mutate(p: Params, rate: float = 0.2) -> Params:
    def flip(cur, opts):
        return random.choice([o for o in opts if o != cur]) if random.random() < rate else cur

    return Params(
        lookbacks=flip(p.lookbacks, [(84, 21, 7), (63, 21, 7), (126, 42, 21)]),
        weights=flip(p.weights, [(0.4, 0.4, 0.2), (0.5, 0.4, 0.1), (0.6, 0.3, 0.1)]),
        skip_recent=flip(p.skip_recent, [0, 21]),
        rebalance=flip(p.rebalance, ["D", "W", "2W", "M"]),
        n_positions_base=1,
        allow_two_when_spread=flip(p.allow_two_when_spread, [True, False]),
        spread_threshold=flip(p.spread_threshold, [0.01, 0.015, 0.02, 0.03]),
        abs_mom_gate_base=flip(p.abs_mom_gate_base, [0.005, 0.01, 0.015]),
        abs_mom_gate_scale_vol=flip(p.abs_mom_gate_scale_vol, [True, False]),
        max_weight_min=flip(p.max_weight_min, [0.7, 0.8]),
        max_weight_max=flip(p.max_weight_max, [0.9, 1.0]),
        cash_buffer_min=flip(p.cash_buffer_min, [0.05, 0.08]),
        cash_buffer_max=flip(p.cash_buffer_max, [0.15, 0.20]),
        use_regime=flip(p.use_regime, [True, False]),
        slow_down_in_chop=flip(p.slow_down_in_chop, [True, False]),
        defensive_override=flip(p.defensive_override, [True, False]),
        defensive_asset=flip(p.defensive_asset, ["TLT", "IEF", "GLD", "IAU", "SHY"]),
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
        allow_two_when_spread=pick(a.allow_two_when_spread, b.allow_two_when_spread),
        spread_threshold=pick(a.spread_threshold, b.spread_threshold),
        abs_mom_gate_base=pick(a.abs_mom_gate_base, b.abs_mom_gate_base),
        abs_mom_gate_scale_vol=pick(a.abs_mom_gate_scale_vol, b.abs_mom_gate_scale_vol),
        max_weight_min=pick(a.max_weight_min, b.max_weight_min),
        max_weight_max=pick(a.max_weight_max, b.max_weight_max),
        cash_buffer_min=pick(a.cash_buffer_min, b.cash_buffer_min),
        cash_buffer_max=pick(a.cash_buffer_max, b.cash_buffer_max),
        use_regime=pick(a.use_regime, b.use_regime),
        slow_down_in_chop=pick(a.slow_down_in_chop, b.slow_down_in_chop),
        defensive_override=pick(a.defensive_override, b.defensive_override),
        defensive_asset=pick(a.defensive_asset, b.defensive_asset),
        universe_mode=a.universe_mode,
    )


# --------------------------- Evolution Loop ------------------------------ #

def evolutionary_search(
        data: Dict[str, pd.DataFrame],
        population_size: int = 28,
        generations: int = 32,
        elite_frac: float = 0.25,
        mutation_rate: float = 0.25,
        seed: Optional[int] = 42,
) -> Tuple[List[Result], Result]:
    if seed is not None:
        random.seed(seed);
        np.random.seed(seed)
    pop = [random_params() for _ in range(population_size)]
    results: List[Result] = []

    for gen in range(generations):
        gen_results = [evaluate(data, p) for p in pop]
        gen_results.sort(key=lambda r: fitness(r), reverse=True)
        best = gen_results[0]
        print(
            f"[Gen {gen + 1}/{generations}] Score={fitness(best):.2f} | "
            f"CAGR={best.stats['CAGR']:.2%} vs VOO {best.bench['CAGR']:.2%} | "
            f"DD={best.stats['MaxDD']:.2%} | N={best.params.n_positions_base}"
            f" | LBs={best.params.lookbacks} | Wts={best.params.weights} | AbsMom>{best.params.abs_mom_gate_base:.2%}"
            f" | Reb={best.params.rebalance} | MaxW=[{best.params.max_weight_min:.2f},{best.params.max_weight_max:.2f}]"
            f" | Spread>{best.params.spread_threshold:.2%} | Regime={'on' if best.params.use_regime else 'off'}"
            f" | Def={'on' if best.params.defensive_override else 'off'}:{best.params.defensive_asset}"
        )
        results.extend(gen_results)

        # selection: elitism + mating pool
        n_elite = max(1, int(elite_frac * population_size))
        elite_params = [r.params for r in gen_results[:n_elite]]

        children: List[Params] = []
        while len(children) < population_size - n_elite:
            pa = random.choice(elite_params)
            pb = random.choice(gen_results[:max(n_elite * 3, n_elite + 3)]).params
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
            "spread_threshold": r.params.spread_threshold,
            "abs_mom_gate_base": r.params.abs_mom_gate_base,
            "abs_mom_gate_scale_vol": r.params.abs_mom_gate_scale_vol,
            "max_weight_min": r.params.max_weight_min,
            "max_weight_max": r.params.max_weight_max,
            "cash_buffer_min": r.params.cash_buffer_min,
            "cash_buffer_max": r.params.cash_buffer_max,
            "use_regime": r.params.use_regime,
            "slow_down_in_chop": r.params.slow_down_in_chop,
            "defensive_override": r.params.defensive_override,
            "defensive_asset": r.params.defensive_asset,
        })
        df = pd.DataFrame(rows).sort_values("score", ascending=False).head(150)
        df.to_csv(f"{OUT_DIR}/optimizer_results_ame_{tag}.csv", index=False)

        manifest = {
            "best_params": {
                "lookbacks": best.params.lookbacks,
                "weights": best.params.weights,
                "skip_recent": best.params.skip_recent,
                "rebalance": best.params.rebalance,
                "n_positions_base": best.params.n_positions_base,
                "allow_two_when_spread": best.params.allow_two_when_spread,
                "spread_threshold": best.params.spread_threshold,
                "abs_mom_gate_base": best.params.abs_mom_gate_base,
                "abs_mom_gate_scale_vol": best.params.abs_mom_gate_scale_vol,
                "max_weight_min": best.params.max_weight_min,
                "max_weight_max": best.params.max_weight_max,
                "cash_buffer_min": best.params.cash_buffer_min,
                "cash_buffer_max": best.params.cash_buffer_max,
                "use_regime": best.params.use_regime,
                "slow_down_in_chop": best.params.slow_down_in_chop,
                "defensive_override": best.params.defensive_override,
                "defensive_asset": best.params.defensive_asset,
            },
            "best_stats": best.stats,
            "best_benchmark": best.bench,
        }
    with open(f"{OUT_DIR}/optimizer_manifest_ame_{tag}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # -------------------------------- Main ----------------------------------- #


def main():
    universe = _dedupe(UNIVERSE)
    print(f"Downloading {len(universe)} tickers...")
    data = fetch_data(universe)
    print(f"Universe after hygiene: {len(data)} tickers")

    all_results, best = evolutionary_search(
        data,
        population_size=28,
        generations=32,
        elite_frac=0.25,
        mutation_rate=0.25,
        seed=42,
    )

    tag = "v1"
    save_results(all_results, best, tag)

    print("\nBest Configuration:")
    print(best.params)
    print("Strategy Stats:")
    for k, v in best.stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")
    print("Benchmark (VOO) Stats:")
    for k, v in best.bench.items():
        print(f"{k}: {v:.4%}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")
    print(f"Score: {fitness(best):.2f}")

    # Optional: plot best run vs VOO
    try:
        eq, _ = run_strategy(data, best.params)
        import matplotlib.pyplot as plt
        plt.figure()
        eq[["equity", "benchmark"]].plot()
        plt.title("AME-V1 Best Strategy vs VOO")
        plt.xlabel("Date");
        plt.ylabel("Equity")
        plt.tight_layout()
        fp = f"{OUT_DIR}/optimizer_ame_best_equity.png"
        plt.savefig(fp);
        plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
