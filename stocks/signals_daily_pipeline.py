# file: signals_daily_pipeline.py
# Purpose: cron-friendly, auto-rolling daily signal generator (JSON/CSV) for AME_V3 strategy
# Python 3.10+
# Deps: pandas, numpy, yfinance
# Usage (cron):  15 22 * * 1-5 /opt/miniconda3/envs/trading3/bin/python /Users/danielrna/IdeaProjects/trading3/signals_daily_pipeline.py

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---- Strategy module import (uses your refactored AME_V3 engine) ----
# Must be in PYTHONPATH or same directory.
import etf_cross_sectional_momentum_v2 as strat


# ----------------------------- IO Helpers -------------------------------- #

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def today_ymd() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y-%m-%d")

def write_json(fp: str, obj: dict) -> None:
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)

def write_csv(fp: str, rows: List[Dict[str, object]], cols: List[str]) -> None:
    pd.DataFrame(rows, columns=cols).to_csv(fp, index=False)


# --------------------------- Signal Generation --------------------------- #

def pick_universe() -> List[str]:
    syms = strat._dedupe(
        strat.UNIVERSE_BASE if strat.UNIVERSE_MODE.upper() == "BASE"
        else (strat.UNIVERSE_BASE + strat.UNIVERSE_EXT)
    )
    return syms

def compute_momentum_panel(close: pd.DataFrame) -> pd.DataFrame:
    return strat.composite_momentum(
        close,
        strat.LOOKBACKS,
        strat.LOOKBACK_WEIGHTS,
        strat.SKIP_RECENT_DAYS
    )

def latest_trading_day(index: pd.DatetimeIndex) -> pd.Timestamp:
    # Use the last index element as the reference day
    return pd.Timestamp(index[-1])

def materialize_signal(
        idx: pd.DatetimeIndex,
        close: pd.DataFrame,
        openp: pd.DataFrame,
        mom: pd.DataFrame,
) -> Tuple[pd.Timestamp, Dict[str, float], float, bool, str]:
    """
    Returns:
      - signal_date
      - target_weights {symbol: weight}
      - cash_buffer
      - defensive_applied
      - defensive_symbol_or_empty
    """
    # Daily scheduling → generate signal for last known market day
    sig_dt = latest_trading_day(idx)

    # Compute target weights via engine's selection logic
    target_w, cash_buf = strat.compute_target_weights(sig_dt, close, mom)

    defensive_applied = False
    defensive_symbol = ""

    if not target_w:
        if strat.DEFENSIVE_OVERRIDE and strat.DEFENSIVE_ASSET in close.columns:
            defensive_applied = True
            defensive_symbol = strat.DEFENSIVE_ASSET
            # Allocate everything (minus CASH_BUFFER_MAX) to defensive
            target_w = {strat.DEFENSIVE_ASSET: float(1.0 - strat.CASH_BUFFER_MAX)}
            cash_buf = float(strat.CASH_BUFFER_MAX)
        else:
            # No signal; stay in cash
            target_w = {}
            cash_buf = 1.0

    # Normalize for safety (numerical drift)
    ssum = sum(max(0.0, v) for v in target_w.values())
    if ssum > 0:
        target_w = {k: float(max(0.0, v) / ssum) * (1.0 - cash_buf) for k, v in target_w.items()}

    return sig_dt, target_w, float(cash_buf), defensive_applied, defensive_symbol


# ------------------------------- Pipeline -------------------------------- #

def run_pipeline(out_dir: str, force: bool) -> None:
    ensure_dir(out_dir)
    signals_dir = os.path.join(out_dir, "signals")
    ensure_dir(signals_dir)

    # 1) Fetch fresh data (universe aligns to BASE_SYMBOL calendar)
    universe = pick_universe()
    data = strat.fetch_data(universe)

    # 2) Build OHLC panels and momentum
    idx, close, openp = strat.build_panel(data)
    mom = compute_momentum_panel(close)

    # 3) Compute today's signal
    sig_dt, target_w, cash_buf, def_applied, def_sym = materialize_signal(idx, close, openp, mom)

    # 4) Compose payload
    cfg = strat._config_dict()
    payload = {
        "as_of": str(sig_dt.date()),
        "generated_at": today_ymd(),
        "params": {
            "lookbacks": cfg["LOOKBACKS"],
            "weights": cfg["LOOKBACK_WEIGHTS"],
            "skip_recent_days": cfg["SKIP_RECENT_DAYS"],
            "rebalance": cfg["REBALANCE_FREQ"],
            "abs_mom_gate_base": cfg["ABS_MOM_GATE_BASE"],
            "abs_mom_gate_scale_vol": cfg["ABS_MOM_GATE_SCALE_VOL"],
            "spread_threshold1": cfg["SPREAD_THRESHOLD1"],
            "spread_threshold2": cfg["SPREAD_THRESHOLD2"],
            "max_weight_min": cfg["MAX_WEIGHT_MIN"],
            "max_weight_max": cfg["MAX_WEIGHT_MAX"],
            "cash_buffer_min": cfg["CASH_BUFFER_MIN"],
            "cash_buffer_max": cfg["CASH_BUFFER_MAX"],
            "defensive_override": cfg["DEFENSIVE_OVERRIDE"],
            "defensive_asset": cfg["DEFENSIVE_ASSET"],
            "universe_mode": cfg["UNIVERSE_MODE"],
        },
        "universe": sorted([s for s in close.columns if s is not None]),
        "signal": {
            "weights": {k: round(float(v), 6) for k, v in sorted(target_w.items(), key=lambda kv: kv[1], reverse=True)},
            "cash_buffer": round(float(cash_buf), 6),
            "defensive_applied": bool(def_applied),
            "defensive_symbol": def_sym,
        },
        "benchmark_mode": cfg["BENCHMARK_MODE"],
    }

    # 5) Idempotency: if same signal already written for the same date, skip unless --force
    signal_key = json.dumps(payload["signal"], sort_keys=True)
    sig_hash = str(hash(signal_key))
    stamp = payload["as_of"]
    out_json = os.path.join(signals_dir, f"signal_{stamp}.json")
    meta_fp = os.path.join(signals_dir, "latest_meta.json")

    prev_hash = None
    if os.path.exists(meta_fp):
        try:
            with open(meta_fp, "r") as f:
                prev = json.load(f)
            if prev.get("as_of") == stamp:
                prev_hash = prev.get("signal_hash")
        except Exception:
            prev_hash = None

    if (prev_hash == sig_hash) and (not force) and os.path.exists(out_json):
        # No-op
        return

    # 6) Write JSON + CSV + meta
    write_json(out_json, payload)

    rows = [{"symbol": sym, "weight": payload["signal"]["weights"].get(sym, 0.0)} for sym in payload["signal"]["weights"].keys()]
    # Include explicit cash row for convenience
    rows.append({"symbol": "CASH", "weight": payload["signal"]["cash_buffer"]})
    out_csv = os.path.join(signals_dir, f"signal_{stamp}.csv")
    write_csv(out_csv, rows, cols=["symbol", "weight"])

    meta = {
        "as_of": stamp,
        "generated_at": payload["generated_at"],
        "files": {"json": out_json, "csv": out_csv},
        "defensive_applied": payload["signal"]["defensive_applied"],
        "defensive_symbol": payload["signal"]["defensive_symbol"],
        "signal_hash": sig_hash,
    }
    write_json(meta_fp, meta)

    # 7) Convenience symlink/copy to latest
    latest_json = os.path.join(signals_dir, "latest.json")
    latest_csv = os.path.join(signals_dir, "latest.csv")
    try:
        if os.path.islink(latest_json):
            os.unlink(latest_json)
        if os.path.islink(latest_csv):
            os.unlink(latest_csv)
    except Exception:
        pass
    try:
        # On platforms without symlink perms, copy instead
        try:
            if os.path.exists(latest_json):
                os.remove(latest_json)
            if os.path.exists(latest_csv):
                os.remove(latest_csv)
            os.symlink(out_json, latest_json)
            os.symlink(out_csv, latest_csv)
        except Exception:
            import shutil
            shutil.copyfile(out_json, latest_json)
            shutil.copyfile(out_csv, latest_csv)
    except Exception:
        pass


# --------------------------------- CLI ----------------------------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily auto-rolling signal generator for AME_V3 strategy")
    p.add_argument("--out-dir", type=str, default=strat.OUT_DIR, help="Output directory (default: strategy OUT_DIR)")
    p.add_argument("--force", action="store_true", help="Force write even if signal unchanged for today's date")
    return p.parse_args(argv)

def main() -> None:
    args = parse_args(sys.argv[1:])
    run_pipeline(out_dir=args.out_dir, force=args.force)

if __name__ == "__main__":
    main()
