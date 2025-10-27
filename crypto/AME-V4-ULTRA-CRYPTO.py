#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================
# AME-ULTRA-CRYPTO — Mono-Objective Max CAGR GA
# ==============================================
# Objective: Maximize CAGR only (no Pareto). Aggressive settings.
# - Single-file, full replacement. Python 3.10+
# - Deps: numpy, pandas, yfinance, matplotlib (optional)
# - Early stop on stagnation (gen-level), immediate exit with best result.
# - Daily rebalancing, top-1 concentration, minimal cash buffer, wide momentum ranges.

import os, json, random, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------- Config ---------------------------------- #

BASE_SYMBOL = "BTC-USD"
UNIVERSE = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD",
    "DOGE-USD","TRX-USD","AVAX-USD","MATIC-USD","DOT-USD",
    "LTC-USD","BCH-USD","LINK-USD","ATOM-USD","UNI-USD",
    "XLM-USD","ETC-USD","USDT-USD","USDC-USD",
]
YF_PERIOD = "max"
OUT_DIR = "./out"
START_CAPITAL = 100_000.0

SLIPPAGE_BPS = 10
FEE_BPS = 10
QTY_DECIMALS = 8

MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL   =   500_000

# --------------------------- GA Hyperparams ------------------------------ #

POP_SIZE            = 140
GENERATIONS         = 200
TOURNAMENT_K        = 4
CXPB                = 0.9     # crossover prob
MUTPB               = 0.6     # mutation prob
MUT_RATE_PER_GENE   = 0.25
ETAC                 = 15.0   # SBX crossover distribution index
ETAM                 = 20.0   # polynomial mutation distribution index
EARLYSTOP_PATIENCE  = 18      # generations with no meaningful improvement
MIN_IMPROVE         = 1e-6
SEED                = 42

random.seed(SEED); np.random.seed(SEED)

# ------------------------------- Data ------------------------------------ #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    syms = _dedupe(symbols)
    raw = yf.download(
        syms, period=period, interval="1d",
        group_by="ticker", auto_adjust=False,
        threads=True, progress=False
    )
    data: Dict[str, pd.DataFrame] = {}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        d = df.rename(columns={
            "Open":"open","High":"high","Low":"low","Close":"close",
            "Adj Close":"adj_close","Volume":"volume"
        }).copy()
        d = d.dropna(subset=["open","high","low","close","volume"])
        bad = (d["open"]<=0)|(d["high"]<=0)|(d["low"]<=0)|(d["close"]<=0)|(d["volume"]<0)
        d = d[~bad]
        d.index.name = "date"
        return d[["open","high","low","close","volume"]]

    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        for s in syms:
            if s in top:
                data[s] = clean(raw[s].copy())
    else:
        # single symbol fallback
        data[syms[0]] = clean(raw)

    if BASE_SYMBOL not in data:
        raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo")
    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

    # Soft liquidity screen
    def liq_ok(df: pd.DataFrame) -> pd.Series:
        adv = (df["close"]*df["volume"]).rolling(30, min_periods=30).mean()
        daily = df["close"]*df["volume"]
        return (adv>MIN_DOLLAR_VOL_AVG_30D) & (daily>MIN_DAILY_DOLLAR_VOL)

    keep = {}
    for s, df in data.items():
        m = liq_ok(df)
        if m.sum() >= int(0.6*len(m)):
            keep[s] = df
    if BASE_SYMBOL not in keep:
        keep[BASE_SYMBOL] = data[BASE_SYMBOL]
    return keep

# ---------------------------- Strategy Core ------------------------------ #

@dataclass(frozen=True)
class Params:
    # Lookbacks (descending)
    lb1: int; lb2: int; lb3: int
    # Weights (sum=1, descending)
    w1: float; w2: float; w3: float
    skip_recent: int            # 0..3
    abs_gate: float             # -0.05..0.05
    # Aggressive defaults fixed inside GA for CAGR-only:
    # top_k=1, rebalance='D'
    max_w: float                # 0.7..1.0
    cash_buf: float             # 0.00..0.02

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip)
    ref  = past.shift(lookback)
    return past.divide(ref) - 1.0

def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    # For CAGR-only we fix to 'D', but keep helper generic
    f = (freq or "D").upper()
    s = idx.to_series()
    if f=="M":
        g = s.groupby(idx.to_period("M")).max()
    elif f=="W":
        g = s.groupby(idx.to_period("W")).max()
    elif f=="2W":
        g = s.groupby(idx.to_period("W")).max().iloc[::2]
    elif f=="D":
        g = s
    else:
        raise ValueError("freq must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)

def apply_cost(price: float, side: str) -> float:
    slip = price*(SLIPPAGE_BPS/10_000)
    fee  = price*(FEE_BPS/10_000)
    return price + slip + fee if side=="buy" else price - slip - fee

def round_qty(q: float) -> float:
    if not np.isfinite(q) or q<=0: return 0.0
    return float(np.floor(q*(10**QTY_DECIMALS))/(10**QTY_DECIMALS))

def build_panel(data: Dict[str,pd.DataFrame]) -> Tuple[pd.DatetimeIndex,pd.DataFrame,pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s:df["close"] for s,df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s:df["open"]  for s,df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def run_strategy(data: Dict[str,pd.DataFrame], p: Params) -> pd.DataFrame:
    idx, close, openp = build_panel(data)

    # Momentum score
    m1 = trailing_return(close, p.lb1, p.skip_recent)
    m2 = trailing_return(close, p.lb2, p.skip_recent)
    m3 = trailing_return(close, p.lb3, p.skip_recent)
    mom = p.w1*m1 + p.w2*m2 + p.w3*m3

    # Rebalance calendar (fixed daily for aggression)
    reb_days = last_day_per_period(idx, 'D')

    lag = max(p.lb1, p.lb2, p.lb3) + p.skip_recent + 10
    first_allowed = idx[0] + pd.Timedelta(days=lag)
    reb_days = reb_days[reb_days >= first_allowed]

    cash = START_CAPITAL
    positions: Dict[str, float] = {}
    equity = pd.Series(index=idx, dtype=float)

    cols = list(mom.columns)

    def pv(dt: pd.Timestamp) -> float:
        v = cash
        if positions:
            syms = list(positions.keys())
            px = close.loc[dt, syms]
            v += float(np.nansum(px.values*np.array([positions[s] for s in syms], dtype=float)))
        return v

    for i, dt in enumerate(idx):
        equity.iloc[i] = pv(dt)
        if dt not in reb_days or i+1 >= len(idx): continue

        t_exec   = idx[i+1]
        port_val = pv(dt)
        row      = mom.loc[dt].to_numpy()
        valid    = np.isfinite(row)
        gated    = valid & (row > p.abs_gate)

        if not gated.any():
            # full liquidation
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty>0:
                        cash += qty*apply_cost(opx,"sell")
                positions.clear()
            continue

        # pick top-1
        order = np.argsort(row[gated])[::-1]
        elig_idx = np.where(gated)[0][order]
        leader = cols[elig_idx[0]]

        usable = 1.0 - p.cash_buf
        target_w = {leader: min(p.max_w, usable)}

        # current notionals
        cur_notional = {s: positions.get(s,0.0)*float(close.loc[dt,s]) for s in positions.keys()}
        tgt_notional = {leader: target_w[leader]*port_val}
        all_syms = sorted(set(list(cur_notional.keys())+list(tgt_notional.keys())))

        # sells first
        for s in all_syms:
            opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt = tgt_notional.get(s,0.0); cur = cur_notional.get(s,0.0)
            diff = tgt - cur
            if diff < -1e-8 and s in positions:
                qty = round_qty((-diff)/opx)
                if qty>0:
                    cash += qty*apply_cost(opx,"sell")
                    positions[s] = positions.get(s,0.0) - qty
                    if positions[s] <= 0:
                        positions.pop(s, None)

        # buys
        for s in all_syms:
            opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt = tgt_notional.get(s,0.0); cur = cur_notional.get(s,0.0)
            diff = tgt - cur
            if diff > 1e-8 and cash>0:
                bpx = apply_cost(opx,"buy")
                qty = round_qty(min(diff,cash)/bpx)
                if qty>0:
                    cost = qty*bpx
                    if cost<=cash:
                        cash -= cost
                        positions[s] = positions.get(s,0.0)+qty

    if len(idx)>0:
        equity.iloc[-1] = pv(idx[-1])

    df = pd.DataFrame(index=idx)
    base_close = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    bench = (START_CAPITAL*(base_close/base_close.iloc[0])).reindex(idx).ffill()
    df["equity"]    = equity
    df["benchmark"] = bench
    return df

def compute_stats(equity: pd.Series) -> Dict[str,float]:
    eq = equity.dropna()
    if len(eq)<2:
        return {k: np.nan for k in ["CAGR","TotalReturn","MaxDD","Vol","Sharpe","MAR","MaxDDDuration"]}
    rets = eq.pct_change().iloc[1:]
    total = float(eq.iloc[-1]/eq.iloc[0]-1.0)
    days  = (eq.index[-1]-eq.index[0]).days
    years = days/365.25 if days>0 else np.nan
    cagr  = (1+total)**(1/years)-1 if years and years>0 else np.nan
    dd    = (eq/eq.cummax()-1.0)
    maxdd = float(dd.min()) if len(dd) else np.nan
    vol   = float(rets.std()*np.sqrt(365)) if rets.std()>0 else np.nan
    sharpe= float(rets.mean()/rets.std()*np.sqrt(365)) if rets.std()>0 else np.nan
    mar   = float(cagr/abs(maxdd)) if (isinstance(maxdd,float) and maxdd<0 and not np.isnan(cagr)) else np.nan
    under = dd<0
    max_len=0; cur_len=0
    for flag in under.astype(int).values:
        if flag: cur_len+=1; max_len=max(max_len,cur_len)
        else: cur_len=0
    return {"CAGR":cagr,"TotalReturn":total,"MaxDD":maxdd,"Vol":vol,"Sharpe":sharpe,"MAR":mar,"MaxDDDuration":float(max_len)}

# -------------------------- Encoding / Ops ------------------------------- #
# Genome u in [0,1]^8 → Params
#   [0] lb1 in [10..365]
#   [1] lb2 in [2..lb1-1]
#   [2] lb3 in [0..lb2]
#   [3,4,5] weights → sort desc, normalize to 1
#   [6] skip_recent in {0..3}
#   [7] abs_gate in [-0.05..0.05]
#   max_w in [0.7..1.0] derived from u[3]
#   cash_buf in [0.00..0.02] derived from u[4]

GENOME_DIM = 8

def decode(u: List[float]) -> Params:
    lb1 = int(10 + u[0]*(365-10))
    lb1 = max(10, min(365, lb1))
    lb2_max = max(2, lb1-1)
    lb2 = int(2 + u[1]*(lb2_max-2))
    lb2 = max(2, min(lb2_max, lb2))
    lb3 = int(0 + u[2]*lb2)
    lb3 = max(0, min(lb2, lb3))

    r = sorted([u[3], u[4], u[5]], reverse=True)
    s = r[0]+r[1]+r[2] + 1e-12
    w1, w2, w3 = r[0]/s, r[1]/s, r[2]/s

    skip = int(u[6]*3+0.5)
    gate = -0.05 + u[7]*0.10

    max_w   = 0.7 + 0.3*u[3]
    cashbuf = 0.02*u[4]

    return Params(lb1, lb2, lb3, w1, w2, w3, skip, gate, max_w, cashbuf)

def sbx_crossover(u: List[float], v: List[float], etac: float = ETAC) -> Tuple[List[float], List[float]]:
    n=len(u); c1=u.copy(); c2=v.copy()
    for i in range(n):
        if random.random() < 0.5:
            x1=min(u[i],v[i]); x2=max(u[i],v[i])
            if abs(x1-x2) < 1e-12:
                continue
            rand=random.random()
            beta = 1.0 + 2.0*(x1-0.0)/(x2-x1)
            alpha= 2.0 - beta**-(etac+1)
            if rand <= 1.0/alpha:
                betaq = (rand*alpha)**(1.0/(etac+1))
            else:
                betaq = (1.0/(2.0 - rand*alpha))**(1.0/(etac+1))
            y1 = 0.5*((x1+x2) - betaq*(x2-x1))

            beta = 1.0 + 2.0*(1.0-x2)/(x2-x1)
            alpha= 2.0 - beta**-(etac+1)
            if rand <= 1.0/alpha:
                betaq = (rand*alpha)**(1.0/(etac+1))
            else:
                betaq = (1.0/(2.0 - rand*alpha))**(1.0/(etac+1))
            y2 = 0.5*((x1+x2) + betaq*(x2-x1))
            c1[i]=min(1.0,max(0.0,y1)); c2[i]=min(1.0,max(0.0,y2))
    return c1, c2

def poly_mutation(u: List[float], etam: float = ETAM, p: float = MUT_RATE_PER_GENE) -> List[float]:
    c = u.copy()
    for i in range(len(u)):
        if random.random() < p:
            r = random.random()
            if r < 0.5:
                delta = (2*r)**(1/(etam+1)) - 1
            else:
                delta = 1 - (2*(1-r))**(1/(etam+1))
            c[i] = min(1.0, max(0.0, u[i] + 0.1*delta))
    return c

# ------------------------------ GA Core ---------------------------------- #

def init_population(n: int) -> List[List[float]]:
    return [np.random.rand(GENOME_DIM).tolist() for _ in range(n)]

def evaluate_cagr(data: Dict[str,pd.DataFrame], p: Params) -> Tuple[float, Dict[str,float]]:
    eq = run_strategy(data, p)
    st = compute_stats(eq["equity"])
    cagr = st["CAGR"]
    if not np.isfinite(cagr):
        return -1e12, st
    # Fitness = +CAGR (maximize). We will maximize directly.
    return cagr, st

def tournament(pop: List[List[float]], fits: List[float], k: int = TOURNAMENT_K) -> List[float]:
    cand = random.sample(range(len(pop)), k)
    # pick best by fitness (maximize)
    best_idx = max(cand, key=lambda i: fits[i])
    return pop[best_idx]

def run_ga(data: Dict[str,pd.DataFrame]):
    os.makedirs(OUT_DIR, exist_ok=True)

    pop = init_population(POP_SIZE)
    fits: List[float] = []
    stats_cache: List[Dict[str,float]] = []

    # initial evaluation
    for x in pop:
        p = decode(x)
        f, st = evaluate_cagr(data, p)
        fits.append(f)
        stats_cache.append(st)

    best_idx = int(np.argmax(fits))
    best_fit = fits[best_idx]
    best_vec = pop[best_idx][:]
    best_params = decode(best_vec)
    best_stats = stats_cache[best_idx]

    stall = 0
    for gen in range(GENERATIONS):
        children=[]
        while len(children) < POP_SIZE:
            p1 = tournament(pop, fits)
            p2 = tournament(pop, fits)
            c1, c2 = p1[:], p2[:]
            if random.random() < CXPB:
                c1, c2 = sbx_crossover(p1, p2)
            if random.random() < MUTPB:
                c1 = poly_mutation(c1)
            if random.random() < MUTPB:
                c2 = poly_mutation(c2)
            children.extend([c1, c2])
        children = children[:POP_SIZE]

        child_fits=[]; child_stats=[]
        for x in children:
            p = decode(x)
            f, st = evaluate_cagr(data, p)
            child_fits.append(f); child_stats.append(st)

        # (μ+λ) selection: combine and take top POP_SIZE
        comb_pop  = pop + children
        comb_fits = fits + child_fits
        comb_stats= stats_cache + child_stats

        order = np.argsort(comb_fits)[::-1]  # descending
        new_pop=[]; new_fits=[]; new_stats=[]
        for i in order[:POP_SIZE]:
            new_pop.append(comb_pop[i]); new_fits.append(comb_fits[i]); new_stats.append(comb_stats[i])
        pop, fits, stats_cache = new_pop, new_fits, new_stats

        gen_best_idx = 0
        gen_best_fit = fits[gen_best_idx]

        improved = gen_best_fit > best_fit + MIN_IMPROVE
        if improved:
            best_fit = gen_best_fit
            best_vec = pop[gen_best_idx][:]
            best_params = decode(best_vec)
            best_stats  = stats_cache[gen_best_idx]
            stall = 0
        else:
            stall += 1

        def pct(x):
            return f"{x*100:.2f}%" if np.isfinite(x) else "nan"

        print(
            f"[Gen {gen+1}/{GENERATIONS}] "
            f"CAGR={pct(best_stats.get('CAGR', np.nan))} | "
            f"DD={pct(best_stats.get('MaxDD', np.nan))} | "
            f"MAR={pct(best_stats.get('MAR', np.nan))} | "
            f"stall={stall}"
        )

        if stall >= EARLYSTOP_PATIENCE:
            print("[EarlyStop] Stagnation detected. Exiting with current best.")
            break

    # Save best artifacts
    manifest = {
        "best_params": {
            "lb1": best_params.lb1, "lb2": best_params.lb2, "lb3": best_params.lb3,
            "w1": best_params.w1, "w2": best_params.w2, "w3": best_params.w3,
            "skip_recent": best_params.skip_recent,
            "abs_gate": best_params.abs_gate,
            "top_k": 1,
            "rebalance": "D",
            "max_w": best_params.max_w,
            "cash_buf": best_params.cash_buf
        },
        "best_stats": best_stats,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/best_cagr_ga.json","w") as f:
        json.dump(manifest, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        eq = run_strategy(data, best_params)
        eq[["equity","benchmark"]].plot()
        plt.title("Max-CAGR GA — Best Strategy vs BTC-USD")
        plt.tight_layout()
        fp = f"{OUT_DIR}/best_cagr_ga_equity.png"
        plt.savefig(fp); plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # Console dump
    print("\nBest Configuration:")
    print(best_params)
    print("Strategy Stats:")
    for k, v in best_stats.items():
        if isinstance(v, float) and np.isfinite(v):
            if k in ("CAGR","TotalReturn","MaxDD","Vol","Sharpe","MAR"):
                # Keep % for these metrics except Sharpe which is unitless, but we still print as % for consistency.
                print(f"{k}: {v:.4%}")
            else:
                print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")

# --------------------------------- Main ---------------------------------- #

def main():
    print(f"Downloading {len(UNIVERSE)} tickers...")
    data = fetch_data(UNIVERSE)
    print(f"Universe after hygiene: {len(data)} tickers")
    run_ga(data)

if __name__ == "__main__":
    main()