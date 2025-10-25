# file: optimizer_momentum_AME_V3.py
# Python 3.10+
# Deps: pandas, numpy, yfinance, matplotlib (optional)

import json, os, random, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_SYMBOL = "VOO"
YF_PERIOD = "max"
START_CAPITAL = 100_000.0
OUT_DIR = "./out"

UNIVERSE = [
    "VOO","VV","VTI","QQQ","DIA","IWM",
    "XLK","XLC","XLY","XLI","XLF","XLV","XLP","XLU","XLB","XLE","XLRE",
    "SMH","SOXX","IGV","FDN",
    "MTUM","QUAL","VLUE","SIZE",
    "XBI","IBB","ICLN","TAN","PBW","ARKK",
    "IEF","TLT","GLD","IAU","SHY"
]

VIX_TICKER = "^VIX"
SLIPPAGE_BPS = 1
FEE_BPS = 1
QTY_DECIMALS = 4
MIN_DOLLAR_VOL_AVG_30D = 2_000_000
MIN_DAILY_DOLLAR_VOL = 500_000

def _dedupe(seq: List[str]) -> List[str]: return list(dict.fromkeys(seq))

def fetch_data(symbols: List[str], period: str = YF_PERIOD) -> Dict[str, pd.DataFrame]:
    symbols = _dedupe(symbols)
    raw = yf.download(symbols, period=period, interval="1d",
                      group_by="ticker", auto_adjust=False, threads=True, progress=False)
    data: Dict[str, pd.DataFrame] = {}
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        d = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"}).copy()
        d = d.dropna(subset=["open","high","low","close","volume"])
        bad = (d["open"]<=0)|(d["high"]<=0)|(d["low"]<=0)|(d["close"]<=0)|(d["volume"]<0)
        d = d[~bad]; d.index.name = "date"
        return d[["open","high","low","close","volume"]]
    if isinstance(raw.columns, pd.MultiIndex):
        for s in symbols:
            if s in raw.columns.get_level_values(0): data[s] = clean(raw[s].copy())
    else:
        s = symbols[0]; data[s] = clean(raw)
    if BASE_SYMBOL not in data: raise ValueError(f"Missing {BASE_SYMBOL} from Yahoo")
    base_idx = data[BASE_SYMBOL].index
    for s in list(data.keys()): data[s] = data[s].reindex(base_idx).dropna()
    def liq_ok(df: pd.DataFrame) -> pd.Series:
        adv = (df["close"]*df["volume"]).rolling(30, min_periods=30).mean()
        daily = df["close"]*df["volume"]; return (adv>MIN_DOLLAR_VOL_AVG_30D)&(daily>MIN_DAILY_DOLLAR_VOL)
    keep={}
    for s,df in data.items():
        m = liq_ok(df)
        if m.sum()>=int(0.6*len(m)): keep[s]=df
    if BASE_SYMBOL not in keep: keep[BASE_SYMBOL]=data[BASE_SYMBOL]
    return keep

def pct_change_series(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods=periods, fill_method=None)

def trailing_return(close: pd.Series, lookback: int, skip: int) -> pd.Series:
    past = close.shift(skip); ref = past.shift(lookback); return past.divide(ref) - 1.0

def apply_cost(price: float, side: str) -> float:
    slip = price*(SLIPPAGE_BPS/10_000); fee = price*(FEE_BPS/10_000)
    return price+slip+fee if side=="buy" else price-slip-fee

def round_qty(q: float) -> float:
    if not np.isfinite(q) or q<=0: return 0.0
    return float(np.floor(q*(10**QTY_DECIMALS))/(10**QTY_DECIMALS))

def last_day_per_period(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f=freq.upper(); s=idx.to_series()
    if f=="M": g=s.groupby(idx.to_period("M")).max()
    elif f=="W": g=s.groupby(idx.to_period("W")).max()
    elif f=="2W": g=s.groupby(idx.to_period("W")).max().iloc[::2]
    elif f=="D": g=s
    else: raise ValueError("freq must be 'D','W','2W','M'")
    return pd.DatetimeIndex(g.values)

def try_fetch_vix(idx: pd.DatetimeIndex) -> Optional[pd.Series]:
    try:
        vix = yf.download(VIX_TICKER, period=YF_PERIOD, interval="1d", progress=False, auto_adjust=False)
        if vix is None or vix.empty: return None
        c = vix.get("Close");
        if c is None or c.empty: return None
        c.index.name="date"; return c.reindex(idx).ffill()
    except Exception:
        return None

def atr_vol_proxy(voo: pd.DataFrame, window: int = 14) -> pd.Series:
    high,low,close = voo["high"],voo["low"],voo["close"]; prev_close = close.shift(1)
    tr = pd.concat([(high-low),(high-prev_close).abs(),(low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean(); atr_pct = atr/close
    return (atr_pct*100.0).reindex(voo.index)

def compute_regime_series(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = data[BASE_SYMBOL].index; out = pd.DataFrame(index=idx)
    vix = try_fetch_vix(idx)
    vol = (atr_vol_proxy(data[BASE_SYMBOL])*100 if vix is None else vix).reindex(idx).ffill().squeeze()
    out["vol_level"]=vol
    vol_s = vol.rolling(10, min_periods=5).mean().squeeze()
    vol_slope = vol_s.pct_change(5).fillna(0.0).squeeze()
    close = data[BASE_SYMBOL]["close"].reindex(idx).squeeze()
    ma200 = close.rolling(200, min_periods=200).mean().squeeze()
    dist = ((close/ma200)-1.0).fillna(0.0).squeeze()
    risk_on = ((dist>0)&(vol_slope<=0)).astype(int)
    risk_off = ((dist<0)&(vol_slope>0)).astype(int)
    chop = (~((risk_on.astype(bool))|(risk_off.astype(bool)))).astype(int)
    out["risk_on"]=risk_on; out["risk_off"]=risk_off; out["chop"]=chop
    out["neg_dist"]=(dist<-0.02).astype(int)  # persistent negative distance flag
    return out

@dataclass(frozen=True)
class Params:
    lookbacks: Tuple[int,int,int,int]     # e.g. (42,14,7,5)
    weights: Tuple[float,float,float,float]  # sum ~1.0
    skip_recent: int
    rebalance: str                        # 'D','W','2W','M'
    n_positions_base: int
    allow_two_when_spread: bool
    allow_three_when_spread: bool
    spread_threshold1: float              # for #2
    spread_threshold2: float              # for #3
    abs_mom_gate_base: float
    abs_mom_gate_scale_vol: bool
    max_weight_min: float
    max_weight_max: float
    max_weight_max_extra: float           # boost cap when spread big (<=1.15)
    cash_buffer_min: float
    cash_buffer_max: float
    use_regime: bool
    slow_down_in_chop: bool
    defensive_override: bool
    defensive_asset: str
    persistent_risk_off_days: int         # trigger defense only if streak reached
    hold_until_decay: bool                # smarter exits
    persistence_bonus_cycles: int         # cycles top1 to get boost
    persistence_boost_cap: float          # +cap (e.g. +0.10)
    universe_mode: str

@dataclass
class Result:
    params: Params
    stats: Dict[str, float]
    bench: Dict[str, float]

def build_panel(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DatetimeIndex,pd.DataFrame,pd.DataFrame]:
    idx = data[BASE_SYMBOL].index
    close = pd.DataFrame({s:df["close"] for s,df in data.items()}, index=idx).dropna(how="all")
    openp = pd.DataFrame({s:df["open"] for s,df in data.items()}, index=idx).reindex(close.index)
    return close.index, close, openp

def momentum_score(close: pd.DataFrame, p: Params) -> pd.DataFrame:
    l1,l2,l3,l4 = p.lookbacks; w1,w2,w3,w4 = p.weights
    m1 = trailing_return(close, l1, p.skip_recent)
    m2 = trailing_return(close, l2, p.skip_recent)
    m3 = trailing_return(close, l3, p.skip_recent)
    m4 = trailing_return(close, l4, p.skip_recent)
    # Activate ultra-short tier only if above cross-sectional median (per date)
    med = m4.median(axis=1).reindex(m4.index).fillna(0.0)
    mask4 = (m4.gt(med, axis=0)).astype(float)
    m4a = m4*mask4
    return w1*m1 + w2*m2 + w3*m3 + w4*m4a

def compute_abs_gate(gate_base: float, vol_level: float, scale: bool) -> float:
    if not scale or not np.isfinite(vol_level): return gate_base
    lo,hi=20.0,35.0
    mult = 1.0 if vol_level<=lo else (1.5 if vol_level>=hi else 1.0+0.5*(vol_level-lo)/(hi-lo))
    return gate_base*mult

def dynamic_max_weight(p: Params, spread: float) -> Tuple[float,float]:
    spread = max(0.0, min(0.10, spread)); frac = spread/0.10
    base_cap = p.max_weight_min + (p.max_weight_max - p.max_weight_min)*frac
    extra_cap = p.max_weight_max_extra if spread>=0.02 else p.max_weight_max
    return base_cap, min(extra_cap, 1.15)

def dynamic_cash_buffer(p: Params, spread: float) -> float:
    spread = max(0.0, min(0.10, spread)); frac = spread/0.10
    return p.cash_buffer_max - (p.cash_buffer_max - p.cash_buffer_min)*frac

def run_strategy(data: Dict[str,pd.DataFrame], p: Params) -> Tuple[pd.DataFrame,pd.DataFrame]:
    idx, close, openp = build_panel(data)
    mom = momentum_score(close, p)
    regime = compute_regime_series(data) if p.use_regime else None

    def effective_reb_days():
        if not p.use_regime or not p.slow_down_in_chop: return last_day_per_period(idx, p.rebalance)
        daily = last_day_per_period(idx, "D")
        if regime is None: return daily
        mask = np.ones(len(daily), dtype=bool)
        chop_days = regime.loc[daily, "chop"].fillna(0).astype(int).values
        counter=0
        for i in range(len(daily)):
            if chop_days[i]==1: mask[i]=(counter%10==0); counter+=1
            else: mask[i]=True
        return pd.DatetimeIndex(daily[mask])

    reb_days = effective_reb_days()
    largest_lb = max(p.lookbacks)+p.skip_recent+10
    first_allowed = idx[0]+pd.Timedelta(days=largest_lb)
    reb_days = reb_days[reb_days>=first_allowed]

    cash = START_CAPITAL
    positions: Dict[str,float] = {}
    equity = pd.Series(index=idx, dtype=float)

    top1_prev: Optional[str] = None
    top1_streak: Dict[str,int] = {}
    risk_off_streak = 0

    defensive_set = {"TLT","IEF","GLD","IAU","SHY"}

    def pv(dt: pd.Timestamp) -> float:
        v=cash
        for s,qty in positions.items():
            if s in close.columns: v += qty*float(close.loc[dt,s])
        return v

    for i,dt in enumerate(idx):
        equity.iloc[i]=pv(dt)
        if dt not in reb_days or i+1>=len(idx):
            # update risk_off streak when using regime
            if p.use_regime:
                if int(regime.loc[dt,"risk_off"])==1 and int(regime.loc[dt,"neg_dist"])==1: risk_off_streak+=1
                else: risk_off_streak=0
            continue

        t_exec = idx[i+1]; port_val = pv(dt)
        vol_level = float(regime.loc[dt,"vol_level"]) if regime is not None else np.nan
        gate = compute_abs_gate(p.abs_mom_gate_base, vol_level, p.abs_mom_gate_scale_vol)

        # Defensive only if persistent risk-off streak reached
        if p.use_regime and p.defensive_override and risk_off_streak>=p.persistent_risk_off_days:
            asset = p.defensive_asset if p.defensive_asset in close.columns else ("SHY" if "SHY" in close.columns else BASE_SYMBOL)
            opx = float(openp.loc[t_exec, asset]) if asset in openp.columns else np.nan
            if np.isfinite(opx):
                cash_buffer = p.cash_buffer_max
                investable = port_val*(1.0-cash_buffer)
                qty = round_qty(investable/opx)
                for s,qty_old in list(positions.items()):
                    if s==asset: continue
                    opx_s = float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                    if np.isfinite(opx_s) and qty_old>0:
                        cash += qty_old*apply_cost(opx_s,"sell"); positions.pop(s,None)
                if qty>0:
                    existing = positions.get(asset,0.0); add = max(0.0, qty-existing)
                    cost = add*apply_cost(opx,"buy")
                    if cost<=cash: cash-=cost; positions[asset]=existing+add
            # streak continues; skip momentum rebalance
            continue

        # eligible set (exclude defensives)
        scores_today = mom.loc[dt].copy()
        elig=[]
        for s in close.columns:
            if s in defensive_set: continue
            sc = float(scores_today.get(s, np.nan))
            if not np.isfinite(sc) or sc<=gate: continue
            elig.append((s, sc))
        if not elig:
            # smart exits: if no one passes, trim losers only (if hold_until_decay True)
            if not p.hold_until_decay:
                for s,qty in list(positions.items()):
                    opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                    if np.isfinite(opx) and qty>0:
                        cash += qty*apply_cost(opx,"sell")
                positions.clear()
            else:
                # sell any current with score < gate and rank >3
                ranked = sorted([(s,float(scores_today.get(s,np.nan))) for s in positions.keys() if np.isfinite(scores_today.get(s,np.nan))],
                                key=lambda x:x[1], reverse=True)
                for rank,(s,sc) in enumerate(ranked, start=1):
                    if sc<gate and rank>3:
                        opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                        if np.isfinite(opx):
                            qty=positions.get(s,0.0)
                            if qty>0: cash += qty*apply_cost(opx,"sell"); positions.pop(s,None)
            risk_off_streak=0
            top1_prev=None
            continue

        elig.sort(key=lambda x:x[1], reverse=True)
        # spread tiers for 2nd and 3rd
        choose=[elig[0][0]]
        spread1 = elig[0][1] - (elig[1][1] if len(elig)>=2 else -1e9)
        spread2 = (elig[1][1]-elig[2][1]) if len(elig)>=3 else -1e9
        if p.allow_two_when_spread and len(elig)>=2 and spread1>=p.spread_threshold1: choose.append(elig[1][0])
        if p.allow_three_when_spread and len(elig)>=3 and spread2>=p.spread_threshold2: choose.append(elig[2][0])

        # persistence boost for top1 cap
        top1_cur = elig[0][0]
        if top1_prev==top1_cur: top1_streak[top1_cur]=top1_streak.get(top1_cur,1)+1
        else: top1_streak[top1_cur]=1
        top1_prev = top1_cur

        # dynamic sizing
        maxw_base, maxw_extra = dynamic_max_weight(p, spread1)
        # apply persistence cap boost if streak condition met
        cap_boost = p.persistence_boost_cap if top1_streak.get(top1_cur,0)>=p.persistence_bonus_cycles else 0.0
        max_cap = min(1.15, max(maxw_extra, p.max_weight_max) + cap_boost)
        cash_buf = dynamic_cash_buffer(p, spread1)

        # target weights
        if len(choose)==1:
            target_w={choose[0]:min(max_cap,1.0-cash_buf)}
        else:
            raw={choose[0]:1.0}
            if len(choose)>=2: raw[choose[1]]=0.85
            if len(choose)>=3: raw[choose[2]]=0.70
            ssum=sum(raw.values()); raw={k:v/ssum for k,v in raw.items()}
            capped={k:min(max_cap,v) for k,v in raw.items()}
            ssum=sum(capped.values());
            target_w = {k:(v/ssum)*(1.0-cash_buf) for k,v in capped.items()} if ssum>0 else {choose[0]:1.0-cash_buf}

        target_notional = {s:target_w[s]*port_val for s in target_w}
        cur_notional = {s:positions.get(s,0.0)*float(close.loc[dt,s]) for s in positions.keys()}
        all_syms = sorted(set(list(cur_notional.keys())+list(target_notional.keys())))

        # smart exits if hold_until_decay: keep holdings that still rank <=3 or pass gate
        if p.hold_until_decay:
            rank_map = {sym:i+1 for i,(sym,_) in enumerate(elig)}
            for s in list(positions.keys()):
                sc = float(scores_today.get(s, np.nan))
                r = rank_map.get(s, 999)
                if (not np.isfinite(sc)) or (sc<gate and r>3 and s not in target_notional):
                    opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                    if np.isfinite(opx):
                        qty=positions.get(s,0.0)
                        if qty>0: cash+=qty*apply_cost(opx,"sell"); positions.pop(s,None)

        # sells
        for s in all_syms:
            if s not in target_notional and s in positions:
                opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
                if np.isfinite(opx):
                    qty=positions.get(s,0.0)
                    if qty>0: cash+=qty*apply_cost(opx,"sell"); positions.pop(s,None)
                continue
            opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt=target_notional.get(s,0.0); cur=cur_notional.get(s,0.0); diff=tgt-cur
            if diff<-1e-8 and s in positions:
                qty=round_qty((-diff)/opx)
                if qty>0:
                    cash+=qty*apply_cost(opx,"sell"); positions[s]=positions.get(s,0.0)-qty
                    if positions[s]<=0: positions.pop(s,None)

        # buys
        for s in all_syms:
            opx=float(openp.loc[t_exec,s]) if s in openp.columns else np.nan
            if not np.isfinite(opx): continue
            tgt=target_notional.get(s,0.0); cur=cur_notional.get(s,0.0); diff=tgt-cur
            if diff>1e-8 and cash>0:
                exec_px=apply_cost(opx,"buy"); qty=round_qty(min(diff,cash)/exec_px)
                if qty>0:
                    cost=qty*exec_px
                    if cost<=cash: cash-=cost; positions[s]=positions.get(s,0.0)+qty

        # update streak counters post-trade
        if p.use_regime:
            if int(regime.loc[dt,"risk_off"])==1 and int(regime.loc[dt,"neg_dist"])==1: risk_off_streak+=1
            else: risk_off_streak=0

    if len(idx)>0: equity.iloc[-1]=pv(idx[-1])
    equity_df = equity.to_frame("equity")
    voo_close = data[BASE_SYMBOL]["close"].reindex(idx).dropna()
    voo_equity = (START_CAPITAL*(voo_close/voo_close.iloc[0])).reindex(idx).ffill()
    equity_df["benchmark"]=voo_equity
    return equity_df, close

def compute_stats(equity: pd.Series) -> Dict[str,float]:
    eq=equity.dropna()
    if len(eq)<2: return {k:np.nan for k in ["CAGR","TotalReturn","MaxDD","Vol","Sharpe","MAR"]}
    rets=eq.pct_change().iloc[1:]; total=float(eq.iloc[-1]/eq.iloc[0]-1.0)
    days=(eq.index[-1]-eq.index[0]).days; years=days/365.25 if days>0 else np.nan
    cagr=(1+total)**(1/years)-1 if years and years>0 else np.nan
    dd=(eq/eq.cummax()-1.0); maxdd=float(dd.min()) if len(dd) else np.nan
    vol=float(rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else np.nan
    mar=float(cagr/abs(maxdd)) if (isinstance(maxdd,float) and maxdd<0 and cagr is not None and not np.isnan(cagr)) else np.nan
    return {"CAGR":cagr,"TotalReturn":total,"MaxDD":maxdd,"Vol":vol,"Sharpe":sharpe,"MAR":mar}

def fitness(res: Result) -> float:
    cagr=res.stats.get("CAGR",np.nan); dd=res.stats.get("MaxDD",np.nan); mar=res.stats.get("MAR",np.nan)
    bench_cagr=res.bench.get("CAGR",np.nan)
    if any(np.isnan(x) for x in [cagr,dd,bench_cagr]): return -1e9
    edge=cagr-bench_cagr; pen=0.0
    if dd<-0.30: pen+=(abs(dd)-0.30)*6.0
    return edge*120.0 + (mar if not np.isnan(mar) else 0.0)*6.0 - pen

def evaluate(data: Dict[str,pd.DataFrame], p: Params) -> Result:
    eq,_=run_strategy(data,p); stats=compute_stats(eq["equity"]); bench=compute_stats(eq["benchmark"])
    return Result(params=p, stats=stats, bench=bench)

def random_params() -> Params:
    return Params(
        lookbacks=random.choice([(84,21,7,5),(63,21,7,5),(42,14,7,5),(126,42,21,7)]),
        weights=random.choice([(0.40,0.35,0.15,0.10),(0.40,0.40,0.15,0.05),(0.45,0.35,0.15,0.05)]),
        skip_recent=random.choice([0,5,7]),
        rebalance=random.choice(["D","W","2W"]),
        n_positions_base=1,
        allow_two_when_spread=True,
        allow_three_when_spread=True,
        spread_threshold1=random.choice([0.005,0.0075,0.01,0.015]),
        spread_threshold2=random.choice([0.005,0.0075,0.01]),
        abs_mom_gate_base=random.choice([0.0025,0.003,0.004,0.005]),
        abs_mom_gate_scale_vol=random.choice([True,False]),
        max_weight_min=random.choice([0.75,0.80]),
        max_weight_max=random.choice([0.90,1.00]),
        max_weight_max_extra=random.choice([1.00,1.05,1.10]),
        cash_buffer_min=random.choice([0.01,0.02,0.03]),
        cash_buffer_max=random.choice([0.08,0.10,0.12]),
        use_regime=random.choice([True,False]),
        slow_down_in_chop=random.choice([True,False]),
        defensive_override=True,
        defensive_asset=random.choice(["TLT","IEF","GLD","IAU","SHY"]),
        persistent_risk_off_days=random.choice([3,5,7]),
        hold_until_decay=True,
        persistence_bonus_cycles=random.choice([2,3,4]),
        persistence_boost_cap=random.choice([0.05,0.08,0.10]),
        universe_mode="EXT"
    )

def mutate(p: Params, rate: float = 0.25) -> Params:
    def flip(cur, opts):
        return random.choice([o for o in opts if o!=cur]) if random.random()<rate else cur
    return Params(
        lookbacks=flip(p.lookbacks, [(84,21,7,5),(63,21,7,5),(42,14,7,5),(126,42,21,7)]),
        weights=flip(p.weights, [(0.40,0.35,0.15,0.10),(0.40,0.40,0.15,0.05),(0.45,0.35,0.15,0.05)]),
        skip_recent=flip(p.skip_recent,[0,5,7]),
        rebalance=flip(p.rebalance,["D","W","2W"]),
        n_positions_base=1,
        allow_two_when_spread=flip(p.allow_two_when_spread,[True,False]),
        allow_three_when_spread=flip(p.allow_three_when_spread,[True,False]),
        spread_threshold1=flip(p.spread_threshold1,[0.005,0.0075,0.01,0.015]),
        spread_threshold2=flip(p.spread_threshold2,[0.005,0.0075,0.01]),
        abs_mom_gate_base=flip(p.abs_mom_gate_base,[0.0025,0.003,0.004,0.005]),
        abs_mom_gate_scale_vol=flip(p.abs_mom_gate_scale_vol,[True,False]),
        max_weight_min=flip(p.max_weight_min,[0.75,0.80]),
        max_weight_max=flip(p.max_weight_max,[0.90,1.00]),
        max_weight_max_extra=flip(p.max_weight_max_extra,[1.00,1.05,1.10]),
        cash_buffer_min=flip(p.cash_buffer_min,[0.01,0.02,0.03]),
        cash_buffer_max=flip(p.cash_buffer_max,[0.08,0.10,0.12]),
        use_regime=flip(p.use_regime,[True,False]),
        slow_down_in_chop=flip(p.slow_down_in_chop,[True,False]),
        defensive_override=True,
        defensive_asset=flip(p.defensive_asset,["TLT","IEF","GLD","IAU","SHY"]),
        persistent_risk_off_days=flip(p.persistent_risk_off_days,[3,5,7]),
        hold_until_decay=flip(p.hold_until_decay,[True,False]),
        persistence_bonus_cycles=flip(p.persistence_bonus_cycles,[2,3,4]),
        persistence_boost_cap=flip(p.persistence_boost_cap,[0.05,0.08,0.10]),
        universe_mode=p.universe_mode
    )

def crossover(a: Params, b: Params) -> Params:
    pick=lambda x,y: random.choice([x,y])
    return Params(
        lookbacks=pick(a.lookbacks,b.lookbacks),
        weights=pick(a.weights,b.weights),
        skip_recent=pick(a.skip_recent,b.skip_recent),
        rebalance=pick(a.rebalance,b.rebalance),
        n_positions_base=1,
        allow_two_when_spread=pick(a.allow_two_when_spread,b.allow_two_when_spread),
        allow_three_when_spread=pick(a.allow_three_when_spread,b.allow_three_when_spread),
        spread_threshold1=pick(a.spread_threshold1,b.spread_threshold1),
        spread_threshold2=pick(a.spread_threshold2,b.spread_threshold2),
        abs_mom_gate_base=pick(a.abs_mom_gate_base,b.abs_mom_gate_base),
        abs_mom_gate_scale_vol=pick(a.abs_mom_gate_scale_vol,b.abs_mom_gate_scale_vol),
        max_weight_min=pick(a.max_weight_min,b.max_weight_min),
        max_weight_max=pick(a.max_weight_max,b.max_weight_max),
        max_weight_max_extra=pick(a.max_weight_max_extra,b.max_weight_max_extra),
        cash_buffer_min=pick(a.cash_buffer_min,b.cash_buffer_min),
        cash_buffer_max=pick(a.cash_buffer_max,b.cash_buffer_max),
        use_regime=pick(a.use_regime,b.use_regime),
        slow_down_in_chop=pick(a.slow_down_in_chop,b.slow_down_in_chop),
        defensive_override=True,
        defensive_asset=pick(a.defensive_asset,b.defensive_asset),
        persistent_risk_off_days=pick(a.persistent_risk_off_days,b.persistent_risk_off_days),
        hold_until_decay=pick(a.hold_until_decay,b.hold_until_decay),
        persistence_bonus_cycles=pick(a.persistence_bonus_cycles,b.persistence_bonus_cycles),
        persistence_boost_cap=pick(a.persistence_boost_cap,b.persistence_boost_cap),
        universe_mode=a.universe_mode
    )

def evolutionary_search(
        data: Dict[str,pd.DataFrame],
        population_size: int = 32,
        generations: int = 48,
        elite_frac: float = 0.25,
        mutation_rate: float = 0.25,
        seed: Optional[int] = 42,
) -> Tuple[List[Result], Result]:
    if seed is not None: random.seed(seed); np.random.seed(seed)
    pop=[random_params() for _ in range(population_size)]
    results: List[Result]=[]
    for gen in range(generations):
        gen_results=[evaluate(data,p) for p in pop]
        gen_results.sort(key=lambda r: fitness(r), reverse=True)
        best=gen_results[0]
        print(f"[Gen {gen+1}/{generations}] Score={fitness(best):.2f} | "
              f"CAGR={best.stats['CAGR']:.2%} vs VOO {best.bench['CAGR']:.2%} | "
              f"DD={best.stats['MaxDD']:.2%} | N={best.params.n_positions_base} | "
              f"LBs={best.params.lookbacks} | Wts={best.params.weights} | AbsGate>{best.params.abs_mom_gate_base:.2%} | "
              f"Reb={best.params.rebalance} | MaxW=[{best.params.max_weight_min:.2f},{best.params.max_weight_max:.2f}->{best.params.max_weight_max_extra:.2f}] | "
              f"Spread>{best.params.spread_threshold1:.2%}/{best.params.spread_threshold2:.2%} | "
              f"Regime={'on' if best.params.use_regime else 'off'} | Def={'on' if best.params.defensive_override else 'off'}:{best.params.defensive_asset}")
        results.extend(gen_results)
        n_elite=max(1,int(elite_frac*population_size))
        elite_params=[r.params for r in gen_results[:n_elite]]
        children: List[Params]=[]
        while len(children)<population_size-n_elite:
            pa=random.choice(elite_params)
            pb=random.choice(gen_results[:max(n_elite*3,n_elite+3)]).params
            child=crossover(pa,pb); child=mutate(child,rate=mutation_rate); children.append(child)
        pop=elite_params+children
    results.sort(key=lambda r: fitness(r), reverse=True)
    return results, results[0]

def save_results(all_results: List[Result], best: Result, tag: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows=[]
    for r in all_results:
        rows.append({
            "score":fitness(r),"CAGR":r.stats.get("CAGR"),"TotalReturn":r.stats.get("TotalReturn"),
            "MaxDD":r.stats.get("MaxDD"),"Sharpe":r.stats.get("Sharpe"),"MAR":r.stats.get("MAR"),
            "Bench_CAGR":r.bench.get("CAGR"),
            "lookbacks":r.params.lookbacks,"weights":r.params.weights,"skip_recent":r.params.skip_recent,
            "rebalance":r.params.rebalance,"n_positions_base":r.params.n_positions_base,
            "allow_two_when_spread":r.params.allow_two_when_spread,"allow_three_when_spread":r.params.allow_three_when_spread,
            "spread_threshold1":r.params.spread_threshold1,"spread_threshold2":r.params.spread_threshold2,
            "abs_mom_gate_base":r.params.abs_mom_gate_base,"abs_mom_gate_scale_vol":r.params.abs_mom_gate_scale_vol,
            "max_weight_min":r.params.max_weight_min,"max_weight_max":r.params.max_weight_max,"max_weight_max_extra":r.params.max_weight_max_extra,
            "cash_buffer_min":r.params.cash_buffer_min,"cash_buffer_max":r.params.cash_buffer_max,
            "use_regime":r.params.use_regime,"slow_down_in_chop":r.params.slow_down_in_chop,
            "defensive_override":r.params.defensive_override,"defensive_asset":r.params.defensive_asset,
            "persistent_risk_off_days":r.params.persistent_risk_off_days,"hold_until_decay":r.params.hold_until_decay,
            "persistence_bonus_cycles":r.params.persistence_bonus_cycles,"persistence_boost_cap":r.params.persistence_boost_cap
        })
    df=pd.DataFrame(rows).sort_values("score", ascending=False).head(200)
    df.to_csv(f"{OUT_DIR}/optimizer_results_ame_v3_{tag}.csv", index=False)
    manifest={
        "best_params": best.params.__dict__,
        "best_stats": best.stats,
        "best_benchmark": best.bench
    }
    with open(f"{OUT_DIR}/optimizer_manifest_ame_v3_{tag}.json","w") as f: json.dump(manifest,f,indent=2)

def main():
    universe=_dedupe(UNIVERSE)
    print(f"Downloading {len(universe)} tickers...")
    data=fetch_data(universe)
    print(f"Universe after hygiene: {len(data)} tickers")
    all_results,best=evolutionary_search(
        data, population_size=36, generations=48, elite_frac=0.28, mutation_rate=0.28, seed=42
    )
    tag="hv_breakout"
    save_results(all_results,best,tag)
    print("\nBest Configuration:"); print(best.params)
    print("Strategy Stats:");
    for k,v in best.stats.items(): print(f"{k}: {v:.4%}" if isinstance(v,float) and not np.isnan(v) else f"{k}: {v}")
    print("Benchmark (VOO) Stats:")
    for k,v in best.bench.items(): print(f"{k}: {v:.4%}" if isinstance(v,float) and not np.isnan(v) else f"{k}: {v}")
    print(f"Score: {fitness(best):.2f}")
    try:
        eq,_=run_strategy(data,best.params)
        import matplotlib.pyplot as plt
        plt.figure(); eq[["equity","benchmark"]].plot()
        plt.title("AME-V3 Breakout-Forcer vs VOO"); plt.xlabel("Date"); plt.ylabel("Equity"); plt.tight_layout()
        fp=f"{OUT_DIR}/optimizer_ame_v3_best_equity.png"; plt.savefig(fp); plt.close()
        print(f"Saved: {fp}")
    except Exception as e:
        print(f"Plot skipped: {e}")

if __name__=="__main__":
    main()
