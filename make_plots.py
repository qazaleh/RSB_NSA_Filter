# make_plots.py
"""
Generate paper-ready plots from results_summary/summary.csv and save them to results_summary/plots/.

Plots created:
- acc_sum_by_setting.png
- rs_freq_by_setting.png
- rs_rate_given_correct_by_setting.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def make_plots(summary_csv: str, out_dir: str) -> None:
    main_args = ["--summary_csv", summary_csv, "--out_dir", out_dir]
    import sys
    sys.argv = ["make_plots.py"] + main_args
    main()


def _ensure_dir(p: Path) -> None:
    """Create directory if missing."""
    p.mkdir(parents=True, exist_ok=True)


def _load_summary(csv_path: Path) -> pd.DataFrame:
    """Load summary CSV and coerce numeric columns."""
    df = pd.read_csv(csv_path)
    for col in ["acc_sum", "acc_concepts_both", "rs_freq", "rs_rate_given_correct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _bar_plot(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    """Save a simple bar plot for one metric."""
    if metric not in df.columns:
        print(f"[SKIP] Missing column: {metric}")
        return

    plot_df = df[["setting", metric]].dropna()
    if plot_df.empty:
        print(f"[SKIP] No valid data for: {metric}")
        return

    plt.figure()
    plt.bar(plot_df["setting"], plot_df[metric])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, default="results_summary/summary.csv")
    ap.add_argument("--out_dir", type=str, default="results_summary/plots")
    args = ap.parse_args()

    csv_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")

    df = _load_summary(csv_path)

    _bar_plot(df, "acc_sum", out_dir / "acc_sum_by_setting.png", "Sum Accuracy by Setting")
    _bar_plot(df, "rs_freq", out_dir / "rs_freq_by_setting.png", "RS Frequency by Setting")
    _bar_plot(
        df,
        "rs_rate_given_correct",
        out_dir / "rs_rate_given_correct_by_setting.png",
        "RS Rate Given Correct by Setting",
    )


if __name__ == "__main__":
    main()
