# file: crypto_momentum_optimizer.py
# Python 3.10+
# Deps: pandas, numpy, yfinance, itertools, tqdm, crypto_cross_sectional_momentum_v2.py

import itertools
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import crypto_cross_sectional_momentum_v2 as strat


def evaluate_params(params, data):
    strat.LOOKBACKS = params["LOOKBACKS"]
    strat.LOOKBACK_WEIGHTS = params["LOOKBACK_WEIGHTS"]
    strat.REBALANCE_FREQ = params["REBALANCE_FREQ"]
    strat.ABS_MOM_GATE_BASE = params["ABS_MOM_GATE_BASE"]
    strat.SPREAD_THRESHOLD1 = params["SPREAD_THRESHOLD1"]
    strat.SPREAD_THRESHOLD2 = params["SPREAD_THRESHOLD2"]

    equity_df, _, _ = strat.run_backtest(data)
    stats = strat.compute_stats(equity_df["equity"])
    return stats


def main():
    symbols = strat._dedupe(
        strat.UNIVERSE_BASE if strat.UNIVERSE_MODE.upper() == "BASE" else (strat.UNIVERSE_BASE + strat.UNIVERSE_EXT)
    )
    data = strat.fetch_data(symbols)

    lookbacks_grid = [(21, 14, 7), (42, 21, 7), (63, 21, 14), (90, 30, 10)]
    weights_grid = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.6, 0.3, 0.1)]
    rebalance_grid = ["D", "W", "2W", "M"]
    abs_gate_grid = [0.002, 0.003, 0.005, 0.007]
    spread1_grid = [0.005, 0.01, 0.015]
    spread2_grid = [0.003, 0.0075, 0.01]

    combos = list(
        itertools.product(
            lookbacks_grid,
            weights_grid,
            rebalance_grid,
            abs_gate_grid,
            spread1_grid,
            spread2_grid,
        )
    )

    results = []
    print(f"Total combinations: {len(combos)}")

    for (
            lookbacks,
            weights,
            reb_freq,
            abs_gate,
            spread1,
            spread2,
    ) in tqdm(combos, desc="Optimizing"):
        params = {
            "LOOKBACKS": lookbacks,
            "LOOKBACK_WEIGHTS": weights,
            "REBALANCE_FREQ": reb_freq,
            "ABS_MOM_GATE_BASE": abs_gate,
            "SPREAD_THRESHOLD1": spread1,
            "SPREAD_THRESHOLD2": spread2,
        }
        try:
            stats = evaluate_params(params, data)
            results.append(
                {
                    **params,
                    "CAGR": stats.get("CAGR", np.nan),
                    "Sharpe": stats.get("Sharpe", np.nan),
                    "MaxDD": stats.get("MaxDD", np.nan),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(results)
    df = df.sort_values("CAGR", ascending=False).reset_index(drop=True)

    os.makedirs("out", exist_ok=True)
    df.to_csv("out/optimizer_results.csv", index=False)

    best = df.iloc[0].to_dict()
    print("\nBest configuration:")
    for k, v in best.items():
        print(f"{k}: {v}")

    with open("out/best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"\nSaved results to out/optimizer_results.csv and best_params.json at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()