"""Inject synthetic drift to demo the drift detection system.

Usage:
    python -m scripts.simulate_drift --mode demand_shock
    python -m scripts.simulate_drift --mode distribution_shift
    python -m scripts.simulate_drift --mode gradual
"""

import argparse

import pandas as pd

from src.config import FEATURES, HOLDOUT_DAYS
from src.data.loader import load_train_data
from src.features.engineering import build_features, load_category_mappings
from src.monitoring.monitor import monitor_once


def demand_shock(df, items=None, multiplier=1.5, start_date="2017-10-01"):
    """Multiply sales for specific items from a start date onward."""
    d = df.copy()
    mask = d["date"] >= pd.Timestamp(start_date)
    if items is not None:
        mask &= d["item"].isin(items)
    d.loc[mask, "sales"] = (d.loc[mask, "sales"] * multiplier).astype(int)
    return d


def distribution_shift(df, swap_pairs=None, start_date="2017-10-01"):
    """Swap sales between item pairs to shift distribution."""
    d = df.copy()
    if swap_pairs is None:
        swap_pairs = [(1, 50), (2, 49), (3, 48)]

    mask = d["date"] >= pd.Timestamp(start_date)
    for a, b in swap_pairs:
        a_mask = mask & (d["item"] == a)
        b_mask = mask & (d["item"] == b)
        a_sales = d.loc[a_mask, "sales"].values
        b_sales = d.loc[b_mask, "sales"].values
        min_len = min(len(a_sales), len(b_sales))
        d.loc[a_mask, "sales"].iloc[:min_len] = b_sales[:min_len]
        d.loc[b_mask, "sales"].iloc[:min_len] = a_sales[:min_len]
    return d


def gradual_drift(df, items=None, daily_growth=1.002, start_date="2017-07-01"):
    """Apply compounding daily growth to simulate gradual drift."""
    d = df.copy()
    mask = d["date"] >= pd.Timestamp(start_date)
    if items is not None:
        mask &= d["item"].isin(items)
    dates = d.loc[mask, "date"]
    start = pd.Timestamp(start_date)
    days_elapsed = (dates - start).dt.days
    multiplier = daily_growth**days_elapsed
    d.loc[mask, "sales"] = (d.loc[mask, "sales"] * multiplier).astype(int)
    return d


def run_simulation(mode):
    print(f"Loading data and applying '{mode}' drift...")
    df = load_train_data()
    mappings = load_category_mappings()

    if mode == "demand_shock":
        drifted = demand_shock(df, items=[1, 2, 3, 4, 5], multiplier=1.8)
    elif mode == "distribution_shift":
        drifted = distribution_shift(df)
    elif mode == "gradual":
        drifted = gradual_drift(df, items=list(range(1, 11)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("Building features for reference (clean) data...")
    ref = build_features(df, mappings).dropna(subset=FEATURES)
    cutoff = ref["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    ref_window = ref[ref["date"] > cutoff]

    print("Building features for drifted data...")
    cur = build_features(drifted, mappings).dropna(subset=FEATURES)
    cur_window = cur[cur["date"] > cutoff]

    print("Running custom PSI/KS drift detection...")
    report = monitor_once(ref_window, cur_window)

    print(f"\nCustom drift detected: {report['drift_detected']}")
    for feat, vals in report["feature_drift"].items():
        status = "DRIFTED" if vals["drifted"] else "OK"
        print(
            f"  {feat}: PSI={vals['psi']:.4f}, KS p={vals['ks_p_value']:.6f} [{status}]"
        )

    try:
        from src.monitoring.evidently_monitor import generate_drift_report

        print("\nGenerating Evidently AI drift report...")
        ev_report = generate_drift_report(ref_window, cur_window)
        print(f"  Dataset drift: {ev_report['dataset_drift']}")
        print(f"  Drifted columns: {ev_report['n_drifted']}/{ev_report['n_columns']}")
        print(f"  HTML report: {ev_report['report_path']}")

        report["evidently"] = ev_report
    except ImportError:
        print(
            "\nSkipping Evidently (not installed). Install with: pip install evidently"
        )
    except Exception as e:  # noqa: BLE001 -- best-effort demo report, must not crash the simulation
        print(f"\nEvidently report failed: {e}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["demand_shock", "distribution_shift", "gradual"],
        default="demand_shock",
    )
    args = parser.parse_args()
    run_simulation(args.mode)
