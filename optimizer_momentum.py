# file: optimizer_momentum.py
# Python 3.10+
# Deps: pandas, numpy, yfinance, matplotlib (optional)
# Purpose: Evolutionary search over momentum strategy params to maximize return vs VOO

import os
import json
import math
import time
import random
import hashlib
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

# Baseline and extended universes (ticker de-dup guarded later)
UNIVERSE_BASE = [
    "VOO","QQQ","IWM","DIA","XLK","XLF","XLV","XLY","XLP","XLI","XLU","XLE","XLRE","XLB"
]
UNIVERSE_EXTENDED = [
    # growth / tech / semis / software
    "QQQ","XLK","SMH","SOXX","IGV","FDN",
    # biotech
    "XBI","IBB",
    # discretionary / comm services
    "XLY","XLC",
    # clean energy / thematic
    "ICLN","TAN","PBW","ARKK",
    # cyclicals
    "XLE","XLF","XLI","XLB",
    # defensives
    "XLV","XLP","XLU",
    # broad
    "VOO","IWM","DIA"
]

# Trading costs
SLIPPAGE_BPS = 1
FEE_BPS = 1
QTY_DECIMALS = 4

# Liquidity guards (loose; ETFs are liquid)
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
            "Open": "open","High":"high","Low":"low","Close":"close",
            "Adj Close":"adj_close","Volume":"volume"
        }).copy()
        d = d.dropna(subset=["open","high","low","close","volume"])
        bad = (d["open"]<=0)|(d["high"]<=0)|(d["low"]<=0)|(d["close"]<=0)|(d["volume"]<0)
        d = d[~bad]
        d.index.name="date"
        return d[["open","high","low","close","volume"]]

    if isinstance(raw.columns, pd.MultiIndex):
        for s in symbols:
            if s not in raw.columns.get_level_values(0):
                continue
            data[s]=clean(raw[s].copy())
    else:
        s=symbols[0]
        data[s]=clean(raw)

    if BASE_SYMBOL not in data:
        raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo")
    base_idx=data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s]=data[s].reindex(base_idx).dropna()

    # liquidity mask (soft)
    def liq_ok(df: pd.DataFrame) -> pd.Series:
        adv=(df["close"]*df["volume"]).rolling(30, min_periods=30).mean()
        daily=df["close"]*df["volume"]
        return (adv>MIN_DOLLAR_VOL_AVG_30D)&(daily>MIN_DAILY_DOLLAR_VOL)

    # drop tickers that are illiquid for majority of time
    keep={}
    for s,df in data.items():
        m=liq_ok(df)
        if m.sum()>=int(0.6*len(m)):  # at least 60% of days pass
            keep[s]=df
    if BASE_SYMBOL not in keep:
        keep[BASE_SYMBOL]=data[BASE_SYMBOL]
    return keep

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past=close.shift(skip)
    ref=past.shift(lookback)
    return past.divide(ref)-1.0

def realized_vol(close: pd.Series, window: int=20) -> pd.Series:
    r=close.pct_change()
    return r.rolling(window, min_periods=window).std()*np.sqrt(252)

def apply_cost(price: float, side: str) -> float:
    slip=price*(SLIPPAGE_BPS/10_000)
    fee=price*(FEE_BPS/10_000)
    return price+slip+fee if side=="buy" else price-slip-fee

def round_qty(q: float) -> float:
    if not np.isfinite(q) or q<=0: return 0.0
    return float(np.floor(q*(10**QTY_DECIMALS))/(10**QTY_DECIMALS))

def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq.upper()=="M":
        g = idx.to_series().groupby(idx.to_period("M")).max()
    elif freq.upper()=="W":
        g = idx.to_series().groupby(idx.to_period("W")).max()
    elif freq.upper()=="2W":
        weekly_last = idx.to_series().groupby(idx.to_period("W")).max()
        g = weekly_last.iloc[::2]
    else:
        raise ValueError("freq must be 'W','2W','M'")
    return pd.DatetimeIndex(g.values)


# ------------------------ Strategy (param-driven) ------------------------- #

@dataclass(frozen=True)
class Params:
    universe_mode: str          # "BASE" | "EXT"
    n_positions: int            # 1..8
    lookbacks: Tuple[int,int,int]   # e.g. (252,126,63)
    weights: Tuple[float,float,float]  # sum to 1.0
    skip_recent: int            # 0 or 21
    rebalance: str              # "W","2W","M"
    inv_vol: bool               # True/False
    max_weight: float           # 0.25..0.60
    ma_filter: bool             # True/False (price > MA200)

def build_panel(data: Dict[str,pd.DataFrame]) -> Tuple[pd.DatetimeIndex,pd.DataFrame,pd.DataFrame]:
    idx=data[BASE_SYMBOL].index
    close=pd.DataFrame({s:df["close"] for s,df in data.items()}, index=idx).dropna(how="all")
    openp=pd.DataFrame({s:df["open"] for s,df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def momentum_score(close: pd.DataFrame, p: Params) -> pd.DataFrame:
    l1,l2,l3 = p.lookbacks
    w1,w2,w3 = p.weights
    m1 = trailing_return(close, l1, p.skip_recent)
    m2 = trailing_return(close, l2, p.skip_recent)
    m3 = trailing_return(close, l3, p.skip_recent)
    return w1*m1 + w2*m2 + w3*m3

def inv_vol_weights(close: pd.DataFrame, window: int=20) -> pd.DataFrame:
    vol = close.apply(lambda s: realized_vol(s, window))
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = 1.0/vol.replace(0,np.nan)
    return iv

def compute_stats(equity: pd.Series) -> Dict[str,float]:
    eq=equity.dropna()
    if len(eq)<2:
        return {"CAGR":np.nan,"TotalReturn":np.nan,"MaxDD":np.nan,"Vol":np.nan,"Sharpe":np.nan,"MAR":np.nan}
    rets=eq.pct_change().iloc[1:]
    tot=float(eq.iloc[-1]/eq.iloc[0]-1.0)
    days=(eq.index[-1]-eq.index[0]).days
    years=days/365.25 if days>0 else np.nan
    cagr=(1+tot)**(1/years)-1 if years and years>0 else np.nan
    dd=(eq/eq.cummax()-1.0)
    maxdd=float(dd.min()) if len(dd) else np.nan
    vol=float(rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan
    mar=float(cagr/abs(maxdd)) if (isinstance(maxdd,float) and maxdd<0 and cagr is not None and not np.isnan(cagr)) else np.nan
    return {"CAGR":cagr,"TotalReturn":tot,"MaxDD":maxdd,"Vol":vol,"Sharpe":sharpe,"MAR":mar}

def run_strategy(data: Dict[str,pd.DataFrame], p: Params) -> Tuple[pd.DataFrame,pd.DataFrame]:
    # reduce to chosen universe
    if p.universe_mode.upper()=="BASE":
        syms=_dedupe(UNIVERSE_BASE)
    else:
        syms=_dedupe(UNIVERSE_BASE+UNIVERSE_EXTENDED)
    syms=[s for s in syms if s in data]
    if BASE_SYMBOL not in syms: syms=[BASE_SYMBOL]+syms
    data_u={s:data[s] for s in syms}

    idx, close, openp = build_panel(data_u)
    mom = momentum_score(close, p)
    reb_days = last_day_per_period(idx, p.rebalance)
    # need enough history for largest lookback
    earliest = idx[0] + pd.Timedelta(days=max(p.lookbacks)+p.skip_recent+10)
    reb_days = reb_days[reb_days>=earliest]

    if p.inv_vol:
        inv = inv_vol_weights(close, 20)
    else:
        inv = pd.DataFrame(1.0, index=close.index, columns=close.columns)

    ma200 = close.rolling(200, min_periods=200).mean() if p.ma_filter else None

    cash=START_CAPITAL
    positions: Dict[str, float] = {}  # qty
    equity=pd.Series(index=idx, dtype=float)

    def pv(dt: pd.Timestamp) -> float:
        val=cash
        for s,qty in positions.items():
            if s in close.columns:
                px=float(close.loc[dt,s])
                val+=qty*px
        return val

    for i,dt in enumerate(idx):
        equity.iloc[i]=pv(dt)
        if dt in reb_days and i+1<len(idx):
            t_exec=idx[i+1]
            port_val=pv(dt)

            # rank by momentum
            elig=[]
            for s in close.columns:
                sc = mom.loc[dt,s] if s in mom.columns else np.nan
                if not np.isfinite(sc) or sc<=0:  # absolute momentum gate
                    continue
                if p.ma_filter:
                    m = ma200.loc[dt,s] if s in ma200.columns else np.nan
                    if not np.isfinite(m) or close.loc[dt,s] <= m:
                        continue
                elig.append((s, float(sc)))
            if not elig:
                # go to cash (sell all)
                for s,qty in list(positions.items()):
                    opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty>0:
                        exec_px=apply_cost(opx,"sell")
                        cash += qty*exec_px
                positions.clear()
                continue

            elig.sort(key=lambda x:x[1], reverse=True)
            top=[s for s,_ in elig[:p.n_positions]]

            # compute raw weights: inv-vol scaled by positive momentum score
            raw={}
            for s in top:
                iv=float(inv.loc[dt,s]) if s in inv.columns else 1.0
                sc=float(mom.loc[dt,s])
                iv=max(iv, 0.0); sc=max(sc,0.0)
                raw[s]=iv*sc
            if sum(raw.values())<=0:
                # equal-weight fallback
                raw={s:1.0 for s in top}
            # cap weights and renormalize
            w={s:raw[s]/sum(raw.values()) for s in raw}
            w={s:min(p.max_weight, max(0.0, wv)) for s,wv in w.items()}
            ssum=sum(w.values())
            if ssum<=0: w={s:1.0/len(top) for s in top}
            else: w={s:wv/ssum for s,wv in w.items()}

            # target dollar per asset
            target_notional={s: w[s]*port_val for s in w}

            # current notional
            cur_notional={}
            for s,qty in positions.items():
                cur_notional[s]=qty*float(close.loc[dt,s])

            all_syms=sorted(set(list(cur_notional.keys())+list(target_notional.keys())))

            # sells
            for s in all_syms:
                opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                if not np.isfinite(opx): continue
                tgt=target_notional.get(s,0.0)
                cur=cur_notional.get(s,0.0)
                diff=tgt-cur
                if diff< -1e-8 and s in positions:
                    qty=round_qty((-diff)/opx)
                    if qty>0:
                        exec_px=apply_cost(opx,"sell")
                        cash += qty*exec_px
                        positions[s]=positions.get(s,0.0)-qty
                        if positions[s]<=0: positions.pop(s,None)
            # buys
            for s in all_syms:
                opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                if not np.isfinite(opx): continue
                tgt=target_notional.get(s,0.0)
                cur=cur_notional.get(s,0.0) if s in cur_notional else 0.0
                diff=tgt-cur
                if diff>1e-8 and cash>0:
                    exec_px=apply_cost(opx,"buy")
                    qty=round_qty(min(diff, cash)/exec_px)
                    if qty>0:
                        cost=qty*exec_px
                        cash -= cost
                        positions[s]=positions.get(s,0.0)+qty

    if len(idx)>0:
        equity.iloc[-1]=pv(idx[-1])
    equity_df=equity.to_frame("equity")

    # Benchmark: VOO buy&hold aligned
    voo_close=data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    voo_equity=(START_CAPITAL*(voo_close/voo_close.iloc[0])).reindex(idx).ffill()
    equity_df["benchmark"]=voo_equity
    return equity_df, close

# ----------------------- Fitness / Evolutionary -------------------------- #

@dataclass
class Result:
    params: Params
    stats: Dict[str,float]
    bench: Dict[str,float]

def fitness(res: Result) -> float:
    # Primary: CAGR advantage vs VOO; Secondary: penalty for huge DD
    cagr=res.stats["CAGR"]
    bench_cagr=res.bench["CAGR"]
    dd=res.stats["MaxDD"]
    if any(map(lambda x: x is None or (isinstance(x,float) and np.isnan(x)), [cagr,bench_cagr,dd])):
        return -1e9
    edge = cagr - bench_cagr
    penalty = 0.0
    if dd < -0.5: penalty += (abs(dd)-0.5)*2.0  # discourage >50% DD
    return edge*100.0 - penalty  # scale for readability

def evaluate(data: Dict[str,pd.DataFrame], p: Params) -> Result:
    eq, _ = run_strategy(data, p)
    stats = compute_stats(eq["equity"])
    bench = compute_stats(eq["benchmark"])
    return Result(params=p, stats=stats, bench=bench)

def random_params() -> Params:
    universe_mode=random.choice(["BASE","EXT"])
    n_positions=random.choice([1,2,3,4,5])
    lookbacks=random.choice([(252,126,63),(252,126,21),(126,63,21),(189,126,63)])
    weights = random.choice([(0.5,0.3,0.2),(0.4,0.4,0.2),(0.34,0.33,0.33),(0.6,0.3,0.1)])
    skip_recent=random.choice([0,21])
    rebalance=random.choice(["W","2W","M"])
    inv_vol=random.choice([True,False])
    max_weight=random.choice([0.25,0.4,0.6])
    ma_filter=random.choice([True,False])
    return Params(universe_mode, n_positions, lookbacks, weights, skip_recent, rebalance, inv_vol, max_weight, ma_filter)

def mutate(p: Params, rate: float=0.2) -> Params:
    def mchoice(cur, options):
        if random.random()<rate:
            return random.choice([o for o in options if o!=cur]) if len(options)>1 else cur
        return cur
    universe_mode=mchoice(p.universe_mode, ["BASE","EXT"])
    n_positions=mchoice(p.n_positions, [1,2,3,4,5,6])
    lookbacks=mchoice(p.lookbacks, [(252,126,63),(252,126,21),(189,126,63),(126,63,21)])
    weights=mchoice(p.weights, [(0.5,0.3,0.2),(0.45,0.35,0.2),(0.4,0.4,0.2),(0.34,0.33,0.33),(0.6,0.3,0.1)])
    skip_recent=mchoice(p.skip_recent, [0,21])
    rebalance=mchoice(p.rebalance, ["W","2W","M"])
    inv_vol=mchoice(p.inv_vol, [True,False])
    max_weight=mchoice(p.max_weight, [0.25,0.4,0.6])
    ma_filter=mchoice(p.ma_filter, [True,False])
    return Params(universe_mode, n_positions, lookbacks, weights, skip_recent, rebalance, inv_vol, max_weight, ma_filter)

def crossover(a: Params, b: Params) -> Params:
    # uniform crossover
    return Params(
        random.choice([a.universe_mode,b.universe_mode]),
        random.choice([a.n_positions,b.n_positions]),
        random.choice([a.lookbacks,b.lookbacks]),
        random.choice([a.weights,b.weights]),
        random.choice([a.skip_recent,b.skip_recent]),
        random.choice([a.rebalance,b.rebalance]),
        random.choice([a.inv_vol,b.inv_vol]),
        random.choice([a.max_weight,b.max_weight]),
        random.choice([a.ma_filter,b.ma_filter]),
    )

# --------------------------- Evolution Loop ------------------------------ #

def evolutionary_search(
        data: Dict[str,pd.DataFrame],
        population_size: int = 24,
        generations: int = 20,
        elite_frac: float = 0.25,
        mutation_rate: float = 0.2,
        seed: Optional[int] = 42
) -> Tuple[List[Result], Result]:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    pop=[random_params() for _ in range(population_size)]
    results: List[Result]=[]

    for gen in range(generations):
        gen_results=[evaluate(data, p) for p in pop]
        gen_results.sort(key=lambda r: fitness(r), reverse=True)
        best=gen_results[0]
        print(f"[Gen {gen+1}/{generations}] Best edge={fitness(best):.3f} | "
              f"CAGR={best.stats['CAGR']:.2%} vs VOO {best.bench['CAGR']:.2%} | "
              f"DD={best.stats['MaxDD']:.2%} | N={best.params.n_positions} "
              f"| LBs={best.params.lookbacks} | Wts={best.params.weights} "
              f"| Reb={best.params.rebalance} | InvVol={best.params.inv_vol} "
              f"| MaxW={best.params.max_weight} | MAfilter={best.params.ma_filter} | Univ={best.params.universe_mode}")
        results.extend(gen_results)

        # selection: elitism + tournament
        n_elite=max(1,int(elite_frac*population_size))
        elite=[r.params for r in gen_results[:n_elite]]

        # breed
        children=[]
        while len(children) < population_size - n_elite:
            pa=random.choice(elite)
            pb=random.choice(gen_results[:max(n_elite*3, n_elite+3)]).params
            child=crossover(pa,pb)
            child=mutate(child, rate=mutation_rate)
            children.append(child)

        pop = elite + children

    # collect all, sort by fitness
    results.sort(key=lambda r: fitness(r), reverse=True)
    return results, results[0]

# ------------------------------- I/O ------------------------------------- #

def save_results(all_results: List[Result], best: Result) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows=[]
    for r in all_results:
        rows.append({
            "fitness": fitness(r),
            "CAGR": r.stats["CAGR"],
            "TotalReturn": r.stats["TotalReturn"],
            "MaxDD": r.stats["MaxDD"],
            "Sharpe": r.stats["Sharpe"],
            "MAR": r.stats["MAR"],
            "Bench_CAGR": r.bench["CAGR"],
            "Bench_MaxDD": r.bench["MaxDD"],
            "universe": r.params.universe_mode,
            "n_positions": r.params.n_positions,
            "lookbacks": r.params.lookbacks,
            "weights": r.params.weights,
            "skip_recent": r.params.skip_recent,
            "rebalance": r.params.rebalance,
            "inv_vol": r.params.inv_vol,
            "max_weight": r.params.max_weight,
            "ma_filter": r.params.ma_filter,
        })
    df=pd.DataFrame(rows)
    # sort and keep top 100
    df=df.sort_values("fitness", ascending=False).head(100)
    df.to_csv(f"{OUT_DIR}/optimizer_results.csv", index=False)

    manifest={
        "best_params": {
            "universe": best.params.universe_mode,
            "n_positions": best.params.n_positions,
            "lookbacks": best.params.lookbacks,
            "weights": best.params.weights,
            "skip_recent": best.params.skip_recent,
            "rebalance": best.params.rebalance,
            "inv_vol": best.params.inv_vol,
            "max_weight": best.params.max_weight,
            "ma_filter": best.params.ma_filter,
        },
        "best_stats": best.stats,
        "best_benchmark": best.bench,
        "generated_at": pd.Timestamp.utcnow().isoformat()
    }
    with open(f"{OUT_DIR}/optimizer_manifest.json","w") as f:
        json.dump(manifest, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)

# -------------------------------- Main ----------------------------------- #

def main():
    universe=_dedupe(UNIVERSE_BASE+UNIVERSE_EXTENDED)
    print(f"Downloading {len(universe)} tickers...")
    data=fetch_data(universe)
    print(f"Universe after hygiene: {len(data)} tickers")

    all_results, best = evolutionary_search(
        data,
        population_size=24,   # adjust for speed/coverage
        generations=16,       # adjust for depth
        elite_frac=0.25,
        mutation_rate=0.25,
        seed=42
    )
    save_results(all_results, best)

    print("\nBest Configuration:")
    print(best.params)
    print("Strategy Stats:")
    for k,v in best.stats.items():
        print(f"{k}: {v:.4%}" if isinstance(v,float) and not np.isnan(v) else f"{k}: {v}")
    print("Benchmark (VOO) Stats:")
    for k,v in best.bench.items():
        print(f"{k}: {v:.4%}" if isinstance(v,float) and not np.isnan(v) else f"{k}: {v}")
    print(f"\nSaved: {OUT_DIR}/optimizer_results.csv, {OUT_DIR}/optimizer_manifest.json")

    # Optional: plot best run vs VOO
    try:
        eq, _ = run_strategy(data, best.params)
        import matplotlib.pyplot as plt
        plt.figure()
        eq[["equity","benchmark"]].plot()
        plt.title("Best Strategy vs VOO (optimizer)")
        plt.xlabel("Date"); plt.ylabel("Equity")
        plt.tight_layout()
        fp=f"{OUT_DIR}/optimizer_best_equity.png"
        plt.savefig(fp); plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__=="__main__":
    main()
