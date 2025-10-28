# optimizer_ga_momentum_fixed.py
# GA optimizer for a spot crypto momentum strategy — robustified and debugged.
# - Core: numpy, pandas only. Optional yfinance for live fetch.
# - Defensive: detailed tracebacks on failures, robust date handling,
#   guards against inadvertent shadowing, and safer index lookups.
# Usage:
#   - Provide CSVs via `csv_map` or ensure yfinance + internet is available.
#   - Run: python optimizer_ga_momentum_fixed.py
# NOTE: This is a research tool. It does not guarantee future returns.

from __future__ import annotations
import sys
import math
import random
import datetime as _dt
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

# --------- Domain models ---------
@dataclass(frozen=True)
class StrategyParams:
    lookback_days: int
    top_n: int
    rebalance_days: int
    momentum_measure: str  # 'simple_return' or 'risk_adjusted'
    min_weight: float
    max_weight: float
    stop_loss: float  # fraction, e.g., 0.2 -> 20% stop
    take_profit: float  # fraction
    position_size_frac: float  # fraction of portfolio per position max

@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    equity: pd.Series
    cagr: float
    total_return: float
    max_drawdown: float
    annual_volatility: float
    sharpe: float

# --------- Utilities ---------
def ensure_datetime_index(obj):
    """Accept Series/DataFrame. Ensure DatetimeIndex and sorted ascending."""
    if isinstance(obj, pd.Series):
        s = obj.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        return s.sort_index()
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    else:
        raise TypeError("ensure_datetime_index expects pandas Series or DataFrame")

def safe_to_timestamp(value):
    """Convert many date-like inputs to pandas.Timestamp safely."""
    if isinstance(value, pd.Timestamp):
        return value
    if value is None:
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        # if user passed a datetime.date or datetime.datetime, pandas will handle it,
        # but fallback to explicit constructors
        if isinstance(value, _dt.date):
            return pd.Timestamp(value)
        raise

# --------- Data fetching / loading ---------
class PriceStore:
    def __init__(self):
        self.prices: Dict[str, pd.Series] = {}

    def load_from_csv(self, symbol: str, path: str, date_col: str = "Date", close_col: str = "Close"):
        df = pd.read_csv(path)
        if date_col not in df.columns or close_col not in df.columns:
            raise ValueError("CSV must contain Date and Close columns (or specify correct names).")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        series = df[close_col].astype(float).rename(symbol)
        self.prices[symbol] = ensure_datetime_index(series)

    def load_from_yfinance(self, symbol: str, start: _dt.date, end: _dt.date):
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False, interval="1d", auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"No data returned for {symbol}")
        close_series = ensure_datetime_index(df["Close"])
        close_series.name = symbol     # <-- instead of .rename(symbol)
        self.prices[symbol] = close_series


    def to_dataframe(self, symbols: Iterable[str], start: Optional[_dt.date] = None, end: Optional[_dt.date] = None) -> pd.DataFrame:
        frames = []
        for s in symbols:
            if s not in self.prices:
                raise KeyError(f"Symbol {s} not loaded")
            ser = self.prices[s].copy()
            ser.name = s
            frames.append(ser)
        df = pd.concat(frames, axis=1).sort_index()
        # forward fill missing instrument days (assume markets may be closed differently)
        df = df.ffill().dropna(how="all")
        if df.empty:
            raise RuntimeError("Combined price DataFrame is empty after concatenation/ffill.")
        if start is not None:
            start_ts = safe_to_timestamp(start)
            df = df[df.index >= start_ts]
        if end is not None:
            end_ts = safe_to_timestamp(end)
            df = df[df.index <= end_ts]
        if df.empty:
            raise RuntimeError("Price DataFrame empty after applying date range filters.")
        # Ensure daily calendar continuity (fill weekends too to keep indexing consistent)
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_index).ffill().dropna(how="all")
        if df.empty:
            raise RuntimeError("Price DataFrame empty after reindex/ffill.")
        return ensure_datetime_index(df)

# --------- Strategy (momentum ranking) ---------
class MomentumStrategy:
    def __init__(self, params: StrategyParams):
        self.params = params

    def rank_and_weights(self, price_df: pd.DataFrame, date) -> pd.Series:
        """
        date: accepts pd.Timestamp, datetime.date, string (ISO), or an index label present in price_df.
        Returns weights Series indexed by price_df.columns.
        Robust against minor label misalignments: will use nearest previous available date.
        """
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df = ensure_datetime_index(price_df)

        # normalize date to Timestamp
        ts = safe_to_timestamp(date)
        if ts is None:
            return pd.Series(0.0, index=price_df.columns)

        # If exact date not present, find the last index <= ts
        try:
            if ts in price_df.index:
                end_idx = price_df.index.get_loc(ts)
            else:
                # get indexer for previous valid location
                pos = price_df.index.get_indexer([ts], method="ffill")[0]
                if pos == -1:
                    return pd.Series(0.0, index=price_df.columns)
                end_idx = pos
        except Exception:
            # fallback safe behavior
            # convert ts to nearest
            pos = price_df.index.get_indexer([ts], method="ffill")[0]
            if pos == -1:
                return pd.Series(0.0, index=price_df.columns)
            end_idx = pos

        lookback = int(max(1, self.params.lookback_days))
        start_idx = end_idx - lookback + 1
        if start_idx < 0:
            return pd.Series(0.0, index=price_df.columns)

        window = price_df.iloc[start_idx:end_idx+1]
        if window.shape[0] < 2:
            return pd.Series(0.0, index=price_df.columns)

        # compute ranking measure
        if self.params.momentum_measure == "simple_return":
            # ratio between last and first price in window
            with np.errstate(divide='ignore', invalid='ignore'):
                ret = window.iloc[-1] / window.iloc[0] - 1.0
            ranked = ret.sort_values(ascending=False)
        else:
            rets = window.pct_change().dropna()
            if rets.empty:
                return pd.Series(0.0, index=price_df.columns)
            mean = rets.mean()
            vol = rets.std().replace(0.0, 1e-9)
            score = mean / vol
            ranked = score.sort_values(ascending=False)

        top = ranked.iloc[: max(1, int(self.params.top_n))]
        if top.empty:
            return pd.Series(0.0, index=price_df.columns)

        # Equal allocation among top
        weights = pd.Series(0.0, index=price_df.columns, dtype=float)
        per = 1.0 / len(top)
        for sym in top.index:
            weights[sym] = per

        # enforce bounds and per-position cap
        weights = weights.clip(lower=float(self.params.min_weight), upper=float(self.params.max_weight))
        cap = float(self.params.position_size_frac)
        weights = weights.apply(lambda x: min(x, cap))
        total = weights.sum()
        if total > 1.0:
            weights = weights / total
        return weights

# --------- Backtester (vectorized, deterministic) ---------
class Backtester:
    def __init__(self, prices: pd.DataFrame, params: StrategyParams, starting_capital: float = 1.0):
        self.prices = ensure_datetime_index(prices)
        self.params = params
        self.starting_capital = float(starting_capital)

    def run(self) -> BacktestResult:
        p = self.prices.copy()
        dates = p.index
        n = len(dates)
        if n == 0:
            raise RuntimeError("Price DataFrame contains no dates.")
        equity = pd.Series(index=dates, dtype=float)
        cash = float(self.starting_capital)
        holdings = {sym: 0.0 for sym in p.columns}
        strategy = MomentumStrategy(self.params)
        last_rebalance_idx: Optional[int] = None

        # Pre-fill equity[0]
        equity.iloc[0] = cash

        for i in range(n):
            date = dates[i]
            # update portfolio mark-to-market from previous holdings
            if i > 0:
                price_today = p.iloc[i]
                value = sum(holdings[sym] * price_today[sym] for sym in p.columns)
                equity.iloc[i] = cash + value
            else:
                equity.iloc[i] = cash

            need_rebalance = False
            if last_rebalance_idx is None:
                need_rebalance = True
            else:
                days_since = (date - dates[last_rebalance_idx]).days
                if days_since >= max(1, int(self.params.rebalance_days)):
                    need_rebalance = True

            if need_rebalance:
                try:
                    target_weights = strategy.rank_and_weights(p, date)
                except Exception:
                    target_weights = pd.Series(0.0, index=p.columns)
                # enforce index alignment
                if not isinstance(target_weights, pd.Series) or not set(target_weights.index).issubset(set(p.columns)):
                    target_weights = pd.Series(0.0, index=p.columns)

                # stop-loss/take-profit checks relative to entry price approximated at last_rebalance_idx
                if last_rebalance_idx is not None:
                    entry_prices = p.iloc[last_rebalance_idx]
                    cur_prices = p.iloc[i]
                    for sym in p.columns:
                        if holdings[sym] != 0:
                            ep = entry_prices.get(sym, None)
                            cp = cur_prices.get(sym, None)
                            if ep is None or cp is None or ep == 0 or pd.isna(ep) or pd.isna(cp):
                                continue
                            change = (cp / ep) - 1.0
                            if change <= -abs(self.params.stop_loss):
                                # liquidate
                                cash += holdings[sym] * cp
                                holdings[sym] = 0.0
                            elif change >= abs(self.params.take_profit):
                                cash += holdings[sym] * cp
                                holdings[sym] = 0.0

                # compute portfolio total value
                cur_prices = p.iloc[i]
                value = sum(holdings[sym] * cur_prices[sym] for sym in p.columns)
                total_equity = cash + value

                # desired allocation amounts
                allocation = {}
                for sym, w in target_weights.items():
                    w = float(w) if not pd.isna(w) else 0.0
                    if w <= 0:
                        allocation[sym] = 0.0
                        continue
                    alloc_amount = total_equity * w
                    cap_amount = total_equity * float(self.params.position_size_frac)
                    allocation[sym] = min(alloc_amount, cap_amount)

                # execute trades to match allocation (simple discrete units)
                for sym in p.columns:
                    want_amount = allocation.get(sym, 0.0)
                    cp = float(p.iloc[i][sym])
                    if cp == 0 or pd.isna(cp):
                        continue
                    current_value = holdings[sym] * cp
                    delta = want_amount - current_value
                    if abs(delta) < 1e-12:
                        continue
                    if delta > 0:
                        units = delta / cp
                        cost = units * cp
                        if cost > cash:
                            units = cash / cp
                            cost = units * cp
                        holdings[sym] += units
                        cash -= cost
                    else:
                        units = (-delta) / cp
                        units = min(units, holdings[sym])
                        holdings[sym] -= units
                        cash += units * cp

                last_rebalance_idx = i
                # update equity after trades
                value = sum(holdings[sym] * p.iloc[i][sym] for sym in p.columns)
                equity.iloc[i] = cash + value

        equity = equity.ffill().fillna(self.starting_capital)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        days = max( (equity.index[-1] - equity.index[0]).days, 1 )
        years = max(days / 365.25, 1/365.25)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
        rets = equity.pct_change().dropna()
        annual_vol = float(rets.std() * math.sqrt(252)) if not rets.empty else 0.0
        sharpe = float((rets.mean() / rets.std() * math.sqrt(252))) if (not rets.empty and rets.std() != 0) else float("nan")
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

        return BacktestResult(dates=equity.index, equity=equity, cagr=cagr, total_return=total_return,
                              max_drawdown=max_dd, annual_volatility=annual_vol, sharpe=sharpe)

# --------- Genetic Algorithm Optimizer ---------
class GAOptimizer:
    def __init__(self,
                 price_df: pd.DataFrame,
                 symbols: List[str],
                 start_date: Optional[_dt.date],
                 end_date: Optional[_dt.date],
                 population_size: int = 60,
                 generations: int = 60,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 seed: Optional[int] = 42):
        self.price_df = ensure_datetime_index(price_df)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.population_size = max(4, population_size)
        self.generations = max(1, generations)
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _random_params(self) -> StrategyParams:
        lookback_days = int(self.rng.randint(5, 180))
        top_n = int(self.rng.randint(1, max(1, min(len(self.symbols), 10))))
        rebalance_days = int(self.rng.choice([1, 3, 7, 14, 21, 30]))
        momentum_measure = self.rng.choice(["simple_return", "risk_adjusted"])
        min_w = 0.0
        max_w = float(self.rng.uniform(0.05, 0.5))
        stop_loss = round(float(self.rng.uniform(0.05, 0.6)), 3)
        take_profit = round(float(self.rng.uniform(0.05, 2.0)), 3)
        pos_frac = round(float(self.rng.uniform(0.05, 0.5)), 3)
        return StrategyParams(
            lookback_days=lookback_days,
            top_n=top_n,
            rebalance_days=rebalance_days,
            momentum_measure=momentum_measure,
            min_weight=min_w,
            max_weight=max_w,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_frac=pos_frac
        )

    def _mutate(self, parent: StrategyParams) -> StrategyParams:
        def u(v, a, b):
            if isinstance(v, int):
                delta = int(self.rng.gauss(0, max(1, v * 0.1)))
                return int(max(a, min(b, v + delta)))
            else:
                delta = self.rng.gauss(0, max(0.01, abs(v) * 0.1))
                return float(max(a, min(b, v + delta)))

        lookback = u(parent.lookback_days, 3, 365)
        top_n = u(parent.top_n, 1, max(1, min(len(self.symbols), 20)))
        rebalance = parent.rebalance_days if self.rng.random() > 0.3 else int(self.rng.choice([1, 3, 7, 14, 21, 30]))
        momentum = parent.momentum_measure if self.rng.random() > 0.2 else self.rng.choice(["simple_return", "risk_adjusted"])
        max_w = round(float(u(parent.max_weight, 0.05, 1.0)), 3)
        stop = round(float(u(parent.stop_loss, 0.01, 1.0)), 3)
        tp = round(float(u(parent.take_profit, 0.01, 5.0)), 3)
        pos_frac = round(float(u(parent.position_size_frac, 0.01, 1.0)), 3)
        return StrategyParams(
            lookback_days=int(lookback),
            top_n=int(top_n),
            rebalance_days=int(rebalance),
            momentum_measure=momentum,
            min_weight=0.0,
            max_weight=float(max_w),
            stop_loss=float(stop),
            take_profit=float(tp),
            position_size_frac=float(pos_frac)
        )

    def _crossover(self, a: StrategyParams, b: StrategyParams) -> Tuple[StrategyParams, StrategyParams]:
        # produce children by averaging / swapping — then mutate for safety
        child1 = StrategyParams(
            lookback_days=int((a.lookback_days + b.lookback_days) // 2),
            top_n=int((a.top_n + b.top_n) // 2),
            rebalance_days=int((a.rebalance_days + b.rebalance_days) // 2),
            momentum_measure=a.momentum_measure if self.rng.random() < 0.5 else b.momentum_measure,
            min_weight=0.0,
            max_weight=float((a.max_weight + b.max_weight) / 2),
            stop_loss=float((a.stop_loss + b.stop_loss) / 2),
            take_profit=float((a.take_profit + b.take_profit) / 2),
            position_size_frac=float((a.position_size_frac + b.position_size_frac) / 2)
        )
        child2 = StrategyParams(
            lookback_days=int((a.lookback_days + b.lookback_days + 1) // 2),
            top_n=int(max(1, (a.top_n + b.top_n + 1) // 2)),
            rebalance_days=int((a.rebalance_days + b.rebalance_days + 1) // 2),
            momentum_measure=b.momentum_measure if self.rng.random() < 0.5 else a.momentum_measure,
            min_weight=0.0,
            max_weight=float((a.max_weight + b.max_weight) / 2),
            stop_loss=float((a.stop_loss + b.stop_loss) / 2),
            take_profit=float((a.take_profit + b.take_profit) / 2),
            position_size_frac=float((a.position_size_frac + b.position_size_frac) / 2)
        )
        return self._mutate(child1), self._mutate(child2)

    def _fitness(self, params: StrategyParams) -> float:
        bt = Backtester(self.price_df, params, starting_capital=1.0)
        try:
            res = bt.run()
        except Exception:
            return -999.0
        cagr = res.cagr
        dd = abs(res.max_drawdown)
        vol = res.annual_volatility
        score = cagr * 100.0
        score -= 50.0 * dd
        score -= 10.0 * vol
        if not math.isnan(res.sharpe):
            score += max(0.0, res.sharpe) * 5.0
        if res.total_return < 0:
            score -= 100.0
        return float(score)

    def optimize(self) -> List[Tuple[StrategyParams, float]]:
        pop: List[StrategyParams] = [self._random_params() for _ in range(self.population_size)]
        scores: List[float] = [self._fitness(ind) for ind in pop]
        for gen in range(self.generations):
            zipped = list(zip(pop, scores))
            zipped.sort(key=lambda x: x[1], reverse=True)
            elite_count = max(1, int(0.1 * self.population_size))
            new_pop = [zipped[i][0] for i in range(elite_count)]
            while len(new_pop) < self.population_size:
                a = self._tournament(pop, scores)
                b = self._tournament(pop, scores)
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self._crossover(a, b)
                else:
                    c1, c2 = a, b
                if self.rng.random() < self.mutation_rate:
                    c1 = self._mutate(c1)
                if self.rng.random() < self.mutation_rate:
                    c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.population_size:
                    new_pop.append(c2)
            pop = new_pop
            scores = [self._fitness(ind) for ind in pop]
        zipped = list(zip(pop, scores))
        zipped.sort(key=lambda x: x[1], reverse=True)
        return zipped

    def _tournament(self, pop: List[StrategyParams], scores: List[float], k=3) -> StrategyParams:
        indices = self.rng.sample(range(len(pop)), k=min(k, len(pop)))
        best = indices[0]
        for idx in indices:
            if scores[idx] > scores[best]:
                best = idx
        return pop[best]

# --------- Entrypoint and orchestration ---------
def load_price_universe_from_csv_paths(map_symbol_to_csv: Dict[str, str],
                                       start: Optional[_dt.date],
                                       end: Optional[_dt.date]) -> pd.DataFrame:
    store = PriceStore()
    for sym, path in map_symbol_to_csv.items():
        store.load_from_csv(sym, path)
    df = store.to_dataframe(list(map_symbol_to_csv.keys()), start, end)
    return df

def load_price_universe_yahoo(symbols: List[str], start: _dt.date, end: _dt.date) -> pd.DataFrame:
    store = PriceStore()
    for s in symbols:
        store.load_from_yfinance(s, start, end)
    df = store.to_dataframe(symbols, start, end)
    return df

def print_top_results(results: List[Tuple[StrategyParams, float]], price_df: pd.DataFrame, top_k: int = 5):
    for rank, (params, score) in enumerate(results[:top_k], start=1):
        bt = Backtester(price_df, params, starting_capital=1.0).run()
        print(f"Rank {rank}: score={score:.4f}, cagr={bt.cagr:.4%}, total_return={bt.total_return:.2%}, max_dd={bt.max_drawdown:.2%}, vol={bt.annual_volatility:.4f}, sharpe={bt.sharpe:.4f}")
        print(params)
        print("-" * 80)

def main():
    csv_map: Dict[str, str] = {}  # fill to use CSV mode
    yahoo_symbols: List[str] = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "LTC-USD"]
    end = _dt.date.today()
    start = end - _dt.timedelta(days=5*365 + 30)

    try:
        if csv_map:
            price_df = load_price_universe_from_csv_paths(csv_map, start, end)
        else:
            price_df = load_price_universe_yahoo(yahoo_symbols, start, end)
    except Exception as exc:
        # Detailed diagnostic output to locate the failure
        print("Data fetch failed:", exc, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("Provide CSVs via csv_map or install yfinance and allow internet access.", file=sys.stderr)
        sys.exit(1)

    symbols = list(price_df.columns)
    if len(symbols) == 0:
        print("No symbols loaded. Abort.", file=sys.stderr)
        sys.exit(1)

    optimizer = GAOptimizer(price_df=price_df,
                            symbols=symbols,
                            start_date=start,
                            end_date=end,
                            population_size=40,
                            generations=40,
                            crossover_rate=0.8,
                            mutation_rate=0.25,
                            seed=1234)
    results = optimizer.optimize()
    print_top_results(results, price_df, top_k=10)

if __name__ == "__main__":
    main()
