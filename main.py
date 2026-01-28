# main.py
"""
Single entry-point script to reproduce the project experiments.

Behavior:
- Downloads MNIST (.gz) into data/RawData if missing
- Generates datasets if missing (or if --force)
- Trains models if missing (or if --force), unless --skip_train
- Aggregates metrics into results_summary/summary.{csv,json}
- Regenerates plots into results_summary/plots/
- Prints a compact terminal report (baseline vs filtered; GEN vs OOD; clean/noisy if available)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Import plotting function (expects make_plots.py at project root)
from make_plots import make_plots


def run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command and raise on failure."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def npz_exists(data_dir: Path) -> bool:
    """Return True if expected dataset .npz files exist in a directory."""
    return (
        (data_dir / "train.npz").exists()
        and (data_dir / "test_gen.npz").exists()
        and (data_dir / "test_ood.npz").exists()
    )


def ensure_mnist(raw_dir: Path) -> None:
    """Ensure MNIST .gz files exist in raw_dir; download missing files."""
    urls = {
        "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    }

    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in urls.items():
        out = raw_dir / fname
        if not out.exists():
            print(f"[DOWNLOAD] {fname}")
            urllib.request.urlretrieve(url, out)
        else:
            print(f"[OK] Found {fname}")


def print_report(summary: Dict[str, Any]) -> None:
    """Print a compact results report from the aggregated summary dict."""
    def _fmt(d: dict) -> str:
        def _f(x):
            return "NA" if x is None else f"{x:.4f}"
        return (
            f"acc_sum={_f(d.get('acc_sum'))} | "
            f"acc_concepts_both={_f(d.get('acc_concepts_both'))} | "
            f"rs_freq={_f(d.get('rs_freq'))} | "
            f"rs_rate|correct={_f(d.get('rs_rate_given_correct'))}"
        )

    print("\n==================== RESULTS SUMMARY ====================")

    # EvenOdd baseline (clean)
    if "evenodd_baseline_gen" in summary:
        print("EvenOdd   | Baseline | Clean | GEN :", _fmt(summary["evenodd_baseline_gen"]))
    if "evenodd_baseline_ood" in summary:
        print("EvenOdd   | Baseline | Clean | OOD :", _fmt(summary["evenodd_baseline_ood"]))

    # EvenOdd noisy metrics (from metrics_filter.npz keys you already produce)
    if "evenodd_base_gen_noisy" in summary:
        print("EvenOdd   | Baseline | Noisy | GEN :", _fmt(summary["evenodd_base_gen_noisy"]))
    if "evenodd_base_ood_noisy" in summary:
        print("EvenOdd   | Baseline | Noisy | OOD :", _fmt(summary["evenodd_base_ood_noisy"]))
    if "evenodd_filt_gen_noisy" in summary:
        print("EvenOdd   | Filtered | Noisy | GEN :", _fmt(summary["evenodd_filt_gen_noisy"]))
    if "evenodd_filt_ood_noisy" in summary:
        print("EvenOdd   | Filtered | Noisy | OOD :", _fmt(summary["evenodd_filt_ood_noisy"]))

    # LargeSmall baseline (clean)
    if "largesmall_baseline_gen" in summary:
        print("LargeSmall| Baseline | Clean | GEN :", _fmt(summary["largesmall_baseline_gen"]))
    if "largesmall_baseline_ood" in summary:
        print("LargeSmall| Baseline | Clean | OOD :", _fmt(summary["largesmall_baseline_ood"]))

    if "noise_sigma_target" in summary:
        print(f"Noise sigma target: {summary['noise_sigma_target']}")

    print("=========================================================\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/RawData", help="Folder for MNIST .gz files")
    ap.add_argument("--evenodd_dir", type=str, default="data/mnadd_evenodd")
    ap.add_argument("--largesmall_dir", type=str, default="data/mnadd_largesmall")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--results_filter_dir", type=str, default="results_filter")
    ap.add_argument("--results_largesmall_dir", type=str, default="results_largesmall")
    ap.add_argument("--summary_dir", type=str, default="results_summary")

    ap.add_argument("--epochs_baseline", type=int, default=5)
    ap.add_argument("--epochs_filter", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--noise_sigma", type=float, default=0.10, help="Noise sigma used in noisy evaluation (if applicable)")
    ap.add_argument("--skip_train", action="store_true", help="Skip training and only aggregate results + regenerate plots")
    ap.add_argument("--force", action="store_true", help="Recompute datasets/models even if outputs already exist")

    args = ap.parse_args()

    root = Path(".").resolve()
    raw_dir = root / args.raw_dir
    evenodd_dir = root / args.evenodd_dir
    largesmall_dir = root / args.largesmall_dir

    results_dir = root / args.results_dir
    results_filter_dir = root / args.results_filter_dir
    results_largesmall_dir = root / args.results_largesmall_dir
    summary_dir = root / args.summary_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 0) Ensure raw MNIST exists (download if missing)
    ensure_mnist(raw_dir)

    # 1) Generate datasets if missing (or if --force)
    # EvenOdd generator does NOT accept --out_dir (per your argparse help)
    if args.force or (not npz_exists(evenodd_dir)):
        run_cmd([sys.executable, "src/make_mnadd_evenodd.py", "--raw_dir", str(raw_dir)])
    else:
        print(f"[OK] Dataset exists: {evenodd_dir}")

    # LargeSmall generator DOES accept --out_dir (per your working logs)
    if args.force or (not npz_exists(largesmall_dir)):
        run_cmd([sys.executable, "src/make_mnadd_largesmall.py", "--raw_dir", str(raw_dir), "--out_dir", str(largesmall_dir)])
    else:
        print(f"[OK] Dataset exists: {largesmall_dir}")

    # 2) Train baselines if needed (or if --force)
    baseline_ckpt_evenodd = results_dir / "baseline_cnn.pt"
    baseline_ckpt_largesmall = results_largesmall_dir / "baseline_cnn.pt"

    if not args.skip_train:
        if args.force or (not baseline_ckpt_evenodd.exists()):
            run_cmd([
                sys.executable, "src/train_baseline.py",
                "--data_dir", str(evenodd_dir),
                "--out_dir", str(results_dir),
                "--epochs", str(args.epochs_baseline),
                "--batch_size", str(args.batch_size),
            ])
        else:
            print(f"[OK] Baseline checkpoint exists: {baseline_ckpt_evenodd}")

        if args.force or (not baseline_ckpt_largesmall.exists()):
            run_cmd([
                sys.executable, "src/train_baseline.py",
                "--data_dir", str(largesmall_dir),
                "--out_dir", str(results_largesmall_dir),
                "--epochs", str(args.epochs_baseline),
                "--batch_size", str(args.batch_size),
            ])
        else:
            print(f"[OK] Baseline checkpoint exists: {baseline_ckpt_largesmall}")
    else:
        print("[SKIP] Training steps are disabled (--skip_train).")

    # 3) Train filter if needed (or if --force)
    filter_ckpt = results_filter_dir / "filter_mlp.pt"
    if not args.skip_train:
        if args.force or (not filter_ckpt.exists()):
            run_cmd([
                sys.executable, "src/train_filter.py",
                "--data_dir", str(evenodd_dir),
                "--baseline_ckpt", str(baseline_ckpt_evenodd),
                "--out_dir", str(results_filter_dir),
                "--epochs", str(args.epochs_filter),
                "--batch_size", str(max(args.batch_size, 512)),
            ])
        else:
            print(f"[OK] Filter checkpoint exists: {filter_ckpt}")
    else:
        print("[SKIP] Filter training disabled (--skip_train).")

    # 4) Collect metrics
    summary: Dict[str, Any] = {}

    base_evenodd_path = results_dir / "baseline_metrics.npz"
    if base_evenodd_path.exists():
        m = np.load(base_evenodd_path, allow_pickle=True)
        summary["evenodd_baseline_gen"] = m["gen"].item()
        summary["evenodd_baseline_ood"] = m["ood"].item()
    else:
        print(f"[WARN] Missing {base_evenodd_path}")

    # metrics_filter.npz contains noisy keys you already produce:
    # ['base_gen_noisy', 'base_ood_noisy', 'filt_gen_noisy', 'filt_ood_noisy']
    filt_path = results_filter_dir / "metrics_filter.npz"
    if filt_path.exists():
        m = np.load(filt_path, allow_pickle=True)
        for k in m.files:
            summary[f"evenodd_{k}"] = m[k].item()
    else:
        print(f"[WARN] Missing {filt_path}")

    base_ls_path = results_largesmall_dir / "baseline_metrics.npz"
    if base_ls_path.exists():
        m = np.load(base_ls_path, allow_pickle=True)
        summary["largesmall_baseline_gen"] = m["gen"].item()
        summary["largesmall_baseline_ood"] = m["ood"].item()
    else:
        print(f"[WARN] Missing {base_ls_path}")

    summary["noise_sigma_target"] = float(args.noise_sigma)

    # 5) Write summary.json
    json_path = summary_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[OK] Wrote: {json_path}")

    # 6) Write summary.csv
    rows = []

    def add_row(name: str, d: dict) -> None:
        rows.append({
            "setting": name,
            "acc_sum": d.get("acc_sum"),
            "acc_concepts_both": d.get("acc_concepts_both"),
            "rs_freq": d.get("rs_freq"),
            "rs_rate_given_correct": d.get("rs_rate_given_correct"),
        })

    # Clean baseline metrics (EvenOdd + LargeSmall)
    if "evenodd_baseline_gen" in summary:
        add_row("evenodd_baseline_gen_clean", summary["evenodd_baseline_gen"])
    if "evenodd_baseline_ood" in summary:
        add_row("evenodd_baseline_ood_clean", summary["evenodd_baseline_ood"])
    if "largesmall_baseline_gen" in summary:
        add_row("largesmall_baseline_gen_clean", summary["largesmall_baseline_gen"])
    if "largesmall_baseline_ood" in summary:
        add_row("largesmall_baseline_ood_clean", summary["largesmall_baseline_ood"])

    # Noisy baseline/filtered metrics (EvenOdd) if present
    if "evenodd_base_gen_noisy" in summary:
        add_row("evenodd_baseline_gen_noisy", summary["evenodd_base_gen_noisy"])
    if "evenodd_base_ood_noisy" in summary:
        add_row("evenodd_baseline_ood_noisy", summary["evenodd_base_ood_noisy"])
    if "evenodd_filt_gen_noisy" in summary:
        add_row("evenodd_filtered_gen_noisy", summary["evenodd_filt_gen_noisy"])
    if "evenodd_filt_ood_noisy" in summary:
        add_row("evenodd_filtered_ood_noisy", summary["evenodd_filt_ood_noisy"])

    csv_path = summary_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["setting", "acc_sum", "acc_concepts_both", "rs_freq", "rs_rate_given_correct"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote: {csv_path}")

    # 7) Print terminal report
    print_report(summary)

    # 8) Generate plots
    make_plots(str(csv_path), str(summary_dir / "plots"))


if __name__ == "__main__":
    main()
