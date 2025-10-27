#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NSGA-II Pareto Optimizer (MODE A — Ultra Exploration)
# Objectives: maximize CAGR, minimize MaxDD
# Python 3.10+. Deps: numpy, pandas, yfinance, matplotlib (optional)

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

# NSGA-II / Search (ULTRA)
POP_SIZE          = 160
GENERATIONS       = 200
ELITE_FRACTION    = 0.12
TOURNAMENT_K      = 4
CXPB              = 0.9
MUTPB             = 0.55
EARLYSTOP_PATIENCE= 16
HYPERVOL_EPS      = 1e-4
LHS_INIT_FACTOR   = 7
KFOLDS            = 4
SEED              = 42

random.seed(SEED); np.random.seed(SEED)

# ------------------------------- Data ------------------------------------ #

def _dedupe(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    syms = _dedupe(symbols)
    raw = yf.download(syms, period=period, interval="1d",
                      group_by="ticker", auto_adjust=False,
                      threads=True, progress=False)
    data: Dict[str, pd.DataFrame] = {}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        d = df.rename(columns={
            "Open": "open","High":"high","Low":"low","Close":"close",
            "Adj Close":"adj_close","Volume":"volume"}).copy()
        d = d.dropna(subset=["open","high","low","close","volume"])
        bad = (d["open"]<=0)|(d["high"]<=0)|(d["low"]<=0)|(d["close"]<=0)|(d["volume"]<0)
        d = d[~bad]; d.index.name = "date"
        return d[["open","high","low","close","volume"]]

    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        for s in syms:
            if s in top:
                data[s] = clean(raw[s].copy())
    else:
        data[syms[0]] = clean(raw)

    if BASE_SYMBOL not in data: raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo")
    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()):
        data[s] = data[s].reindex(base_idx).dropna()

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
    # momentum
    lb1: int; lb2: int; lb3: int
    w1: float; w2: float; w3: float
    skip_recent: int
    abs_gate: float
    top_k: int
    rebalance: str
    # sizing
    max_w: float
    cash_buf: float
    # volatility targeting
    target_vol: float         # annualized target vol (e.g., 0.8..3.0)
    vol_lb: int               # lookback days for vol
    vol_cap: float            # per-asset vol cap scale
    # regime filter
    regime_sma: int           # SMA length on BTC
    regime_on: int            # 0/1
    bear_mult: float          # scale risk when below SMA
    # crash stop
    crash_dd: float           # 0.3..0.7 → liquidate when DD exceeds
    # survival filter
    min_mom_share: float      # require mom > abs_gate across this share of universe else cash

def trailing_return(close: pd.Series|pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    past = close.shift(skip); ref = past.shift(lookback)
    return past.divide(ref) - 1.0

def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f = (freq or "D").upper(); s = idx.to_series()
    if f=="M": g = s.groupby(idx.to_period("M")).max()
    elif f=="W": g = s.groupby(idx.to_period("W")).max()
    elif f=="2W": g = s.groupby(idx.to_period("W")).max().iloc[::2]
    elif f=="D": g = s
    else: raise ValueError("freq must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)

def apply_cost(price: float, side: str) -> float:
    slip = price*(SLIPPAGE_BPS/10_000); fee = price*(FEE_BPS/10_000)
    return price+slip+fee if side=="buy" else price-slip-fee

def round_qty(q: float) -> float:
    if not np.isfinite(q) or q<=0: return 0.0
    return float(np.floor(q*(10**QTY_DECIMALS))/(10**QTY_DECIMALS))

def build_panel(data: Dict[str,pd.DataFrame]) -> Tuple[pd.DatetimeIndex,pd.DataFrame,pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s:df["close"] for s,df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s:df["open"]  for s,df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def ann_vol(returns: pd.DataFrame, lb: int) -> pd.Series:
    r = returns.rolling(lb, min_periods=max(5, lb//3)).std()
    return r.iloc[-1]*np.sqrt(365)

def run_strategy(data: Dict[str,pd.DataFrame], p: Params) -> pd.DataFrame:
    idx, close, openp = build_panel(data)
    # momentum score
    m1 = trailing_return(close, p.lb1, p.skip_recent)
    m2 = trailing_return(close, p.lb2, p.skip_recent)
    m3 = trailing_return(close, p.lb3, p.skip_recent)
    mom = p.w1*m1 + p.w2*m2 + p.w3*m3

    # daily returns
    rets = close.pct_change(fill_method=None)

    # regime
    btc = close[BASE_SYMBOL]
    sma = btc.rolling(p.regime_sma, min_periods=p.regime_sma//3).mean() if p.regime_on else None
    regime_bear = (btc < sma) if p.regime_on else pd.Series(False, index=idx)

    # rebal calendar
    reb_days = last_day_per_period(idx, p.rebalance)
    lag = max(p.lb1, p.lb2, p.lb3, p.vol_lb, p.regime_sma if p.regime_on else 0) + p.skip_recent + 10
    first_allowed = idx[0] + pd.Timedelta(days=lag)
    reb_days = reb_days[reb_days >= first_allowed]

    cash = START_CAPITAL; positions: Dict[str,float] = {}
    equity = pd.Series(index=idx, dtype=float)
    hwm = START_CAPITAL
    crashed = False

    def pv(dt: pd.Timestamp) -> float:
        v = cash
        if positions:
            syms = list(positions.keys())
            px = close.loc[dt, syms]
            v += float(np.nansum(px.values*np.array([positions[s] for s in syms], dtype=float)))
        return v

    cols = list(mom.columns)

    for i, dt in enumerate(idx):
        cur_val = pv(dt)
        equity.iloc[i] = cur_val
        hwm = max(hwm, cur_val)
        if hwm>0 and (hwm - cur_val)/hwm >= p.crash_dd:
            # Crash stop → liquidate at next open and pause one rebalance
            crashed = True

        if dt not in reb_days or i+1>=len(idx): continue
        t_exec = idx[i+1]; port_val = pv(dt)

        # Crash liquidation branch
        if crashed:
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty>0:
                        cash += qty*apply_cost(opx,"sell")
                positions.clear()
            crashed = False
            continue

        # Market health gate: require cross-sectional momentum breadth
        row = mom.loc[dt].to_numpy(); valid = np.isfinite(row)
        if valid.sum()==0:
            # liquidate
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty>0:
                        cash += qty*apply_cost(opx,"sell")
                positions.clear()
            continue

        gated = valid & (row > p.abs_gate)
        breadth = gated.sum()/max(1, valid.sum())
        if breadth < p.min_mom_share:
            if positions:
                for s, qty in list(positions.items()):
                    opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty>0:
                        cash += qty*apply_cost(opx,"sell")
                positions.clear()
            continue

        order = np.argsort(row[gated])[::-1]
        elig_idx = np.where(gated)[0][order]
        picks = [cols[j] for j in elig_idx[:p.top_k]]

        # target sizing (base)
        usable = 1.0 - p.cash_buf
        if p.top_k==1 or len(picks)==1:
            target = {picks[0]: min(p.max_w, usable)}
        else:
            base = np.linspace(1.0, 0.7, num=len(picks))
            base = base/base.sum()
            cap  = np.minimum(base, p.max_w)
            cap  = cap / cap.sum() * usable
            target = {s: float(w) for s,w in zip(picks, cap)}

        # volatility targeting (per-asset scaling to reach portfolio target)
        sub_rets = rets[picks].loc[:dt].iloc[-(p.vol_lb+2):]  # small window slice
        # compute last-ann-vol per asset
        vol_series = {}
        for s in picks:
            vs = sub_rets[s].rolling(p.vol_lb, min_periods=max(5, p.vol_lb//3)).std().iloc[-1]
            vol_series[s] = float(vs*np.sqrt(365)) if np.isfinite(vs) else np.nan
        # inverse vol weights with cap
        inv = {}
        for s in picks:
            v = vol_series.get(s, np.nan)
            if not np.isfinite(v) or v<=1e-9: v = 1.0
            inv[s] = min(1.0/v, p.vol_cap)
        inv_sum = sum(inv.values()) if inv else 1.0
        ivw = {s: inv[s]/inv_sum for s in picks}

        # blend momentum sizing and inv-vol
        # 70% momentum sizing + 30% inverse-vol → then scale to target_vol
        blended = {}
        for s in picks:
            m_w = target[s]
            v_w = ivw[s]*usable
            blended[s] = 0.7*m_w + 0.3*v_w

        # regime filter scaling
        if p.regime_on and bool(regime_bear.loc[dt]):
            blended = {s: blended[s]*p.bear_mult for s in picks}

        # normalize to usable
        ssum = sum(blended.values())
        if ssum>0:
            blended = {s: blended[s]/ssum * usable for s in picks}
        else:
            blended = {picks[0]: usable}

        # approximate port vol and scale to target_vol
        # proxy: sqrt(sum( w^2 * vol_i^2 )), ignore corr
        port_vol = np.sqrt(sum((blended[s]**2)*(vol_series.get(s,1.0)**2) for s in picks))
        if np.isfinite(port_vol) and port_vol>1e-6:
            scale = min(2.0, max(0.25, p.target_vol/port_vol))
            blended = {s: min(p.max_w, blended[s]*scale) for s in picks}
            ssum = sum(blended.values())
            if ssum>0: blended = {s: blended[s]/ssum * usable for s in picks}

        # current notionals
        cur_notional = {s: positions.get(s,0.0)*float(close.loc[dt,s]) for s in positions.keys()}
        tgt_notional = {s: w*port_val for s,w in blended.items()}
        all_syms = sorted(set(list(cur_notional.keys())+list(tgt_notional.keys())))

        # sells
        for s in all_syms:
            opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt = tgt_notional.get(s,0.0); cur = cur_notional.get(s,0.0)
            diff = tgt-cur
            if diff < -1e-8 and s in positions:
                qty = round_qty((-diff)/opx)
                if qty>0:
                    cash += qty*apply_cost(opx,"sell")
                    positions[s] = positions.get(s,0.0)-qty
                    if positions[s]<=0: positions.pop(s,None)

        # buys
        for s in all_syms:
            opx = float(openp.loc[t_exec, s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt = tgt_notional.get(s,0.0); cur = cur_notional.get(s,0.0)
            diff = tgt-cur
            if diff > 1e-8 and cash>0:
                bpx = apply_cost(opx,"buy")
                qty = round_qty(min(diff,cash)/bpx)
                if qty>0:
                    cost = qty*bpx
                    if cost<=cash:
                        cash -= cost
                        positions[s] = positions.get(s,0.0)+qty

    if len(idx)>0: equity.iloc[-1] = pv(idx[-1])
    df = pd.DataFrame(index=idx)
    base_close = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    bench = (START_CAPITAL*(base_close/base_close.iloc[0])).reindex(idx).ffill()
    df["equity"] = equity
    df["benchmark"] = bench
    return df

def compute_stats(equity: pd.Series) -> Dict[str,float]:
    eq = equity.dropna()
    if len(eq)<2: return {k: np.nan for k in ["CAGR","TotalReturn","MaxDD","Vol","Sharpe","MAR","MaxDDDuration"]}
    rets = eq.pct_change(fill_method=None).iloc[1:]
    total = float(eq.iloc[-1]/eq.iloc[0]-1.0)
    days = (eq.index[-1]-eq.index[0]).days
    years = days/365.25 if days>0 else np.nan
    cagr = (1+total)**(1/years)-1 if years and years>0 else np.nan
    dd = (eq/eq.cummax()-1.0)
    maxdd = float(dd.min()) if len(dd) else np.nan
    vol = float(rets.std()*np.sqrt(365)) if rets.std()>0 else np.nan
    sharpe = float(rets.mean()/rets.std()*np.sqrt(365)) if rets.std()>0 else np.nan
    mar = float(cagr/abs(maxdd)) if (isinstance(maxdd,float) and maxdd<0 and not np.isnan(cagr)) else np.nan
    under = dd<0; max_len=0; cur_len=0
    for flag in under.astype(int).values:
        if flag: cur_len+=1; max_len=max(max_len,cur_len)
        else: cur_len=0
    return {"CAGR":cagr,"TotalReturn":total,"MaxDD":maxdd,"Vol":vol,"Sharpe":sharpe,"MAR":mar,"MaxDDDuration":float(max_len)}

# ---------------------------- Walk-Forward -------------------------------- #

def kfold_time_indices(idx: pd.DatetimeIndex, k: int) -> List[Tuple[pd.Timestamp,pd.Timestamp]]:
    n = len(idx); folds=[]
    seg = n//k
    for i in range(k):
        a = i*seg
        b = (i+1)*seg if i<k-1 else n
        folds.append((idx[a], idx[b-1]))
    return folds

def evaluate_params(data: Dict[str,pd.DataFrame], p: Params) -> Tuple[float,float,float]:
    idx = data[BASE_SYMBOL].index
    folds = kfold_time_indices(idx, KFOLDS)
    cagr_list=[]; dd_list=[]; mar_list=[]
    for (a,b) in folds:
        sub_data = {s: df[(df.index>=a)&(df.index<=b)] for s,df in data.items()}
        if any(len(df)<250 for df in sub_data.values()): continue
        eq = run_strategy(sub_data, p)
        st = compute_stats(eq["equity"])
        if not np.isfinite(st["CAGR"]) or not np.isfinite(st["MaxDD"]): continue
        cagr_list.append(st["CAGR"]); dd_list.append(abs(st["MaxDD"])); mar_list.append(st["MAR"] if np.isfinite(st["MAR"]) else 0.0)
    if not cagr_list: return 1e6, 1e6, -1e6
    c_med = float(np.median(cagr_list)); d_med = float(np.median(dd_list)); m_med=float(np.median(mar_list))
    return -c_med, d_med, m_med

# --------------------------- NSGA-II Toolkit ------------------------------ #

def dominates(a,b):
    return (a[0]<=b[0] and a[1]<=b[1]) and (a[0]<b[0] or a[1]<b[1])

def fast_non_dominated_sort(objs: List[Tuple[float,float]]):
    S=[[] for _ in objs]; n=[0]*len(objs); fronts=[[]]
    for p in range(len(objs)):
        for q in range(len(objs)):
            if dominates(objs[p], objs[q]): S[p].append(q)
            elif dominates(objs[q], objs[p]): n[p]+=1
        if n[p]==0: fronts[0].append(p)
    i=0
    while len(fronts[i])>0:
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0: Q.append(q)
        i+=1; fronts.append(Q)
    fronts.pop()
    return fronts

def crowding_distance(front: List[int], objs: List[Tuple[float,float]]):
    if not front: return {}
    dist={i:0.0 for i in front}
    for m in range(2):
        front_sorted=sorted(front, key=lambda i: objs[i][m])
        dist[front_sorted[0]]=dist[front_sorted[-1]]=float("inf")
        vals=[objs[i][m] for i in front_sorted]
        vmin, vmax = min(vals), max(vals)
        if vmax==vmin: continue
        for j in range(1,len(front_sorted)-1):
            prev=objs[front_sorted[j-1]][m]; nxt=objs[front_sorted[j+1]][m]
            dist[front_sorted[j]] += (nxt-prev)/(vmax-vmin)
    return dist

def tournament(pop, objs, k=TOURNAMENT_K):
    cand = random.sample(range(len(pop)), k)
    fronts = fast_non_dominated_sort([objs[i] for i in cand])
    cd = crowding_distance(fronts[0], [objs[c] for c in cand])
    win = cand[max(fronts[0], key=lambda i: cd[i])]
    return pop[win]

# ----------------------- Encoding / Variation Ops ------------------------ #

REB_CHOICES = ['D','W','2W','M']

def lhs_sample(n, dims):
    cut = np.linspace(0,1,n+1)
    u = np.random.rand(n,dims)
    a = cut[:n]; b = cut[1:n+1]
    rdpoints = u*(b-a)[:,None] + a[:,None]
    H = np.zeros_like(rdpoints)
    for j in range(dims):
        order = np.random.permutation(n)
        H[:,j] = rdpoints[order,j]
    return H

def decode(vec) -> Params:
    # vec in [0,1]^16 (ULTRA)
    lb_max=220
    a1 = int(5 + vec[0]*(lb_max-5))
    a2 = int(2 + vec[1]*max(1,a1-2))
    a3 = int(0 + vec[2]*max(0,a2))
    r = sorted([vec[3],vec[4],vec[5]], reverse=True); s = sum(r)+1e-12
    w1,w2,w3 = r[0]/s, r[1]/s, r[2]/s
    skip = int(vec[6]*3+0.5)
    gate = -0.02 + vec[7]*0.05
    topk = 1 + int(vec[8]*2+0.5)  # 1..3
    reb  = REB_CHOICES[min(int(vec[9]*len(REB_CHOICES)), len(REB_CHOICES)-1)]
    max_w = 0.3 + 0.7*vec[10]
    cash_buf = 0.10*vec[11]
    target_vol = 0.6 + 2.4*vec[12]          # 0.6..3.0
    vol_lb = int(10 + vec[13]*70)           # 10..80
    vol_cap = 0.5 + 2.0*vec[14]             # per-asset inv-vol cap (0.5..2.5)
    regime_sma = int(80 + vec[15]*420)      # 80..500
    regime_on = 1 if vec[5]>0.15 else 0
    bear_mult = 0.1 + 0.9*vec[4]            # 0.1..1.0
    crash_dd = 0.3 + 0.4*vec[7]             # 0.3..0.7
    min_mom_share = 0.05 + 0.35*vec[3]      # 5%..40%
    return Params(a1,a2,a3, w1,w2,w3, skip, gate, topk, reb, max_w, cash_buf,
                  target_vol, vol_lb, vol_cap, regime_sma, regime_on, bear_mult,
                  crash_dd, min_mom_share)

def sbx_crossover(u, v, etac=15.0):
    n=len(u); c1=u.copy(); c2=v.copy()
    for i in range(n):
        if random.random()>0.5:
            x1=min(u[i],v[i]); x2=max(u[i],v[i])
            if abs(x1-x2)<1e-12: continue
            rand=random.random()
            beta=1.0+2.0*(x1-0.0)/(x2-x1)
            alpha=2.0 - beta**-(etac+1)
            if rand<=1.0/alpha: betaq=(rand*alpha)**(1.0/(etac+1))
            else:                betaq=(1.0/(2.0 - rand*alpha))**(1.0/(etac+1))
            nc1=0.5*((x1+x2) - betaq*(x2-x1))
            beta=1.0+2.0*(1.0-x2)/(x2-x1)
            alpha=2.0 - beta**-(etac+1)
            if rand<=1.0/alpha: betaq=(rand*alpha)**(1.0/(etac+1))
            else:                betaq=(1.0/(2.0 - rand*alpha))**(1.0/(etac+1))
            nc2=0.5*((x1+x2) + betaq*(x2-x1))
            c1[i]=max(0.0,min(1.0,nc1)); c2[i]=max(0.0,min(1.0,nc2))
    return c1, c2

def poly_mutation(u, etam=20.0, p=0.25):
    c=u.copy()
    for i in range(len(u)):
        if random.random()<p:
            r=random.random()
            d = (2*r)**(1/(etam+1)) - 1 if r<0.5 else 1 - (2*(1-r))**(1/(etam+1))
            c[i]=min(1.0, max(0.0, u[i]+d*0.12))
    return c

# ----------------------------- Main Search -------------------------------- #

def hypervolume_2d(front: List[Tuple[float,float]], ref=(1e3,1e3)):
    pareto = sorted(front, key=lambda x:(x[0],x[1]))
    hv=0.0; prev_f2=ref[1]
    for f1,f2 in pareto:
        if f2<prev_f2:
            hv += (ref[0]-f1)*(prev_f2-f2)
            prev_f2=f2
    return hv

def nsga2_optimize(data: Dict[str,pd.DataFrame]):
    os.makedirs(OUT_DIR, exist_ok=True)

    # LHS init
    n_init = max(POP_SIZE, LHS_INIT_FACTOR*16)
    H = lhs_sample(n_init, 16)
    pop = [H[i].tolist() for i in range(n_init)]
    random.shuffle(pop)
    pop = pop[:POP_SIZE]

    # Evaluate
    objs=[]; extras=[]
    for x in pop:
        p=decode(x)
        f1,f2,mar = evaluate_params(data,p)
        objs.append((f1,f2)); extras.append(mar)

    best_hv=-float("inf"); stall=0

    for gen in range(GENERATIONS):
        # Non-dominated sorting
        fronts = fast_non_dominated_sort(objs)
        # Selection (elitism)
        new_pop=[]; new_objs=[]; new_extras=[]
        for front in fronts:
            if len(new_pop)+len(front) > POP_SIZE:
                cd = crowding_distance(front, objs)
                order = sorted(front, key=lambda i: cd[i], reverse=True)
                needed = POP_SIZE - len(new_pop)
                for idx in order[:needed]:
                    new_pop.append(pop[idx]); new_objs.append(objs[idx]); new_extras.append(extras[idx])
                break
            else:
                for idx in front:
                    new_pop.append(pop[idx]); new_objs.append(objs[idx]); new_extras.append(extras[idx])
        pop, objs, extras = new_pop, new_objs, new_extras

        # Reproduction
        children=[]
        while len(children)<POP_SIZE:
            p1 = tournament(pop, objs)
            p2 = tournament(pop, objs)
            c1, c2 = p1[:], p2[:]
            if random.random()<CXPB: c1,c2 = sbx_crossover(p1,p2)
            if random.random()<MUTPB: c1 = poly_mutation(c1)
            if random.random()<MUTPB: c2 = poly_mutation(c2)
            children.extend([c1,c2])
        children = children[:POP_SIZE]

        # Evaluate children
        child_objs=[]; child_extras=[]
        for x in children:
            p=decode(x)
            f1,f2,mar = evaluate_params(data,p)
            child_objs.append((f1,f2)); child_extras.append(mar)

        # Combine and select next gen
        comb_pop = pop+children
        comb_objs = objs+child_objs
        comb_extras= extras+child_extras
        fronts = fast_non_dominated_sort(comb_objs)
        next_pop=[]; next_objs=[]; next_extras=[]
        for front in fronts:
            if len(next_pop)+len(front) > POP_SIZE:
                cd = crowding_distance(front, comb_objs)
                order = sorted(front, key=lambda i: cd[i], reverse=True)
                needed = POP_SIZE - len(next_pop)
                for idx in order[:needed]:
                    next_pop.append(comb_pop[idx]); next_objs.append(comb_objs[idx]); next_extras.append(comb_extras[idx])
                break
            else:
                for idx in front:
                    next_pop.append(comb_pop[idx]); next_objs.append(comb_objs[idx]); next_extras.append(comb_extras[idx])
        pop, objs, extras = next_pop, next_objs, next_extras

        # Early stop by hypervolume
        fronts = fast_non_dominated_sort(objs)
        front0 = fronts[0]
        cur_front = [objs[i] for i in front0]
        ref = (max(f for f,_ in cur_front)+1.0, max(s for _,s in cur_front)+1.0)
        hv = hypervolume_2d(cur_front, ref)
        if hv <= best_hv*(1+HYPERVOL_EPS) and hv<=best_hv:
            stall+=1
        else:
            best_hv=hv; stall=0
        print(f"[Gen {gen+1}/{GENERATIONS}] Pareto={len(front0)} | HV={hv:.4f} | stall={stall}")
        if stall>=EARLYSTOP_PATIENCE:
            print("[EarlyStop] Hypervolume stalled. Exiting.")
            break

    # Final Pareto
    fronts = fast_non_dominated_sort(objs)
    pf_idx = fronts[0]
    pareto = []
    for i in pf_idx:
        p = decode(pop[i]); f1,f2 = objs[i]
        pareto.append({"params":p.__dict__, "obj": {"neg_CAGR_med": f1, "MaxDD_med": f2, "MAR_med": extras[i]}})
    pareto_sorted = sorted(pareto, key=lambda r: (r["obj"]["neg_CAGR_med"], r["obj"]["MaxDD_med"]))

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/pareto_nsga2_ultra.json","w") as f:
        json.dump(pareto_sorted, f, indent=2)
    print(f"Saved Pareto: {OUT_DIR}/pareto_nsga2_ultra.json")

    # Pick single best by MAR_med subject to mild MaxDD bound (median)
    cand = sorted(pareto_sorted, key=lambda r: (-r["obj"]["MAR_med"], r["obj"]["MaxDD_med"]))
    best = cand[0] if cand else None
    if best:
        with open(f"{OUT_DIR}/best_nsga2_ultra.json","w") as f:
            json.dump(best, f, indent=2)
        print(f"Saved Best: {OUT_DIR}/best_nsga2_ultra.json")

        # Plot equity (optional)
        try:
            import matplotlib.pyplot as plt
            p = Params(**best["params"])
            eq = run_strategy(data, p)
            eq[["equity","benchmark"]].plot()
            plt.title("Best NSGA-II ULTRA Strategy vs BTC-USD")
            plt.tight_layout()
            fp = f"{OUT_DIR}/best_nsga2_ultra_equity.png"
            plt.savefig(fp); plt.close()
            print(f"Saved: {fp}")
        except Exception as e:
            print(f"Plot skipped: {e}")

# --------------------------------- Main ---------------------------------- #

def main():
    print(f"Downloading {len(UNIVERSE)} tickers...")
    data = fetch_data(UNIVERSE)
    print(f"Universe after hygiene: {len(data)} tickers")
    nsga2_optimize(data)

if __name__ == "__main__":
    main()