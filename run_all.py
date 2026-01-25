# run_all.py
"""
Single entry-point script for reproducing all experiments and results reported in this project:
- generate datasets if missing
- train baseline if missing checkpoint
- train filter if missing checkpoint
- evaluate baseline vs filtered on:
  - MNAdd-EvenOdd: clean GEN/OOD + noisy GEN/OOD
  - MNAdd-LargeSmall: clean GEN/OOD
- single summary CSV/JSON into results_summary/

Assumptions:
- datasets live in:
    data/mnadd_evenodd/{train.npz,test_gen.npz,test_ood.npz}
    data/mnadd_largesmall/{train.npz,test_gen.npz,test_ood.npz}
- scripts exist in src/:
    make_mnadd_evenodd.py
    make_mnadd_largesmall.py
    train_baseline.py
    train_filter.py
- checkpoints default to:
    results/baseline_cnn.pt
    results_filter/filter_mlp.pt
    results_largesmall/baseline_cnn.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

from pathlib import Path
import urllib.request

def ensure_mnist(raw_dir: Path) -> None:
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



raw_dir = Path("data/RawData")
ensure_mnist(raw_dir)

def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def npz_exists(data_dir: Path) -> bool:
    """Check if expected dataset files exist."""
    return (data_dir / "train.npz").exists() and (data_dir / "test_gen.npz").exists() and (data_dir / "test_ood.npz").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/RawData", help="Folder with MNIST .gz files")
    ap.add_argument("--evenodd_dir", type=str, default="data/mnadd_evenodd")
    ap.add_argument("--largesmall_dir", type=str, default="data/mnadd_largesmall")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--results_filter_dir", type=str, default="results_filter")
    ap.add_argument("--results_largesmall_dir", type=str, default="results_largesmall")
    ap.add_argument("--summary_dir", type=str, default="results_summary")

    ap.add_argument("--epochs_baseline", type=int, default=8)
    ap.add_argument("--epochs_filter", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--noise_sigma", type=float, default=0.10, help="Noise level used in noisy evaluation")
    ap.add_argument("--skip_train", action="store_true", help="Skip training steps and only run evaluations (requires checkpoints)")
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

    # ----------------------------
    # 1) Generate datasets if missing
    # ----------------------------
    if not npz_exists(largesmall_dir):
        run_cmd([sys.executable, "src/make_mnadd_evenodd.py", "--raw_dir", str(raw_dir)])
        run_cmd([sys.executable, "src/make_mnadd_largesmall.py", "--raw_dir", str(raw_dir), "--out_dir", str(largesmall_dir)])
    else:
        print(f"[OK] Dataset exists: {largesmall_dir}")

    # ----------------------------
    # 2) Train baseline(s) if needed
    # ----------------------------
    baseline_ckpt_evenodd = results_dir / "baseline_cnn.pt"
    baseline_ckpt_largesmall = results_largesmall_dir / "baseline_cnn.pt"

    if not args.skip_train:
        if not baseline_ckpt_evenodd.exists():
            run_cmd([
                sys.executable, "src/train_baseline.py",
                "--data_dir", str(evenodd_dir),
                "--out_dir", str(results_dir),
                "--epochs", str(args.epochs_baseline),
                "--batch_size", str(args.batch_size),
            ])
        else:
            print(f"[OK] Baseline checkpoint exists: {baseline_ckpt_evenodd}")

        if not baseline_ckpt_largesmall.exists():
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

    # ----------------------------
    # 3) Train filter if needed (EvenOdd only)
    # ----------------------------
    filter_ckpt = results_filter_dir / "filter_mlp.pt"
    if not args.skip_train:
        if not filter_ckpt.exists():
            run_cmd([
                sys.executable, "src/train_filter.py",
                "--data_dir", str(evenodd_dir),
                "--baseline_ckpt", str(baseline_ckpt_evenodd),
                "--out_dir", str(results_filter_dir),
                "--epochs", str(args.epochs_filter),
                "--batch_size", str(max(args.batch_size, 512)),  # faster on CPU if possible
            ])
        else:
            print(f"[OK] Filter checkpoint exists: {filter_ckpt}")
    else:
        print("[SKIP] Filter training disabled (--skip_train).")

    # ----------------------------
    # 4) Collect metrics files produced by scripts
    # ----------------------------
    # ALl metrics save with scripts:
    # - results/baseline_metrics.npz
    # - results_filter/metrics_filter.npz
    # - results_largesmall/baseline_metrics.npz
    #
     
    # This summary currently includes metrics saved by the training and evaluation scripts.
    # Noisy-evaluation results are reported separately and are not aggregated here.
    summary: Dict[str, Any] = {}

    def load_npz_metrics(path: Path) -> Any:
        obj = np.load(path, allow_pickle=True)
        # saved dicts come back as 0-d object arrays; .item() retrieves the dict
        if obj.files and isinstance(obj[obj.files[0]], np.ndarray) and obj[obj.files[0]].dtype == object:
            # caller handles structure
            return obj
        return obj

    # EvenOdd baseline
    base_evenodd_path = results_dir / "baseline_metrics.npz"
    if base_evenodd_path.exists():
        m = np.load(base_evenodd_path, allow_pickle=True)
        summary["evenodd_baseline_gen"] = m["gen"].item()
        summary["evenodd_baseline_ood"] = m["ood"].item()
    else:
        print(f"[WARN] Missing {base_evenodd_path}")

    # EvenOdd filter (contains baseline + filtered)
    filt_path = results_filter_dir / "metrics_filter.npz"
    if filt_path.exists():
        m = np.load(filt_path, allow_pickle=True)
        for k in m.files:
         summary[f"evenodd_{k}"] = m[k].item()
    else:
        print(f"[WARN] Missing {filt_path}")

    # LargeSmall baseline
    base_ls_path = results_largesmall_dir / "baseline_metrics.npz"
    if base_ls_path.exists():
        m = np.load(base_ls_path, allow_pickle=True)
        summary["largesmall_baseline_gen"] = m["gen"].item()
        summary["largesmall_baseline_ood"] = m["ood"].item()
    else:
        print(f"[WARN] Missing {base_ls_path}")

    summary["noise_sigma_target"] = float(args.noise_sigma)

    # ----------------------------
    # 5) Write summary files
    # ----------------------------
    json_path = summary_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[OK] Wrote: {json_path}")

    # A minimal CSV (flat) for paper tables
    # (just key metrics flattned.)
    rows = []
    def add_row(name: str, d: dict):
        rows.append({
            "setting": name,
            "acc_sum": d.get("acc_sum"),
            "acc_concepts_both": d.get("acc_concepts_both"),
            "rs_freq": d.get("rs_freq"),
            "rs_rate_given_correct": d.get("rs_rate_given_correct"),
        })

    if "evenodd_baseline_gen" in summary: add_row("evenodd_baseline_gen_clean", summary["evenodd_baseline_gen"])
    if "evenodd_baseline_ood" in summary: add_row("evenodd_baseline_ood_clean", summary["evenodd_baseline_ood"])
    if "evenodd_filter_filtered_gen" in summary: add_row("evenodd_filtered_gen_clean", summary["evenodd_filter_filtered_gen"])
    if "evenodd_filter_filtered_ood" in summary: add_row("evenodd_filtered_ood_clean", summary["evenodd_filter_filtered_ood"])
    if "largesmall_baseline_gen" in summary: add_row("largesmall_baseline_gen_clean", summary["largesmall_baseline_gen"])
    if "largesmall_baseline_ood" in summary: add_row("largesmall_baseline_ood_clean", summary["largesmall_baseline_ood"])
    if "evenodd_base_gen_noisy" in summary: add_row("evenodd_baseline_gen_noisy", summary["evenodd_base_gen_noisy"])
    if "evenodd_base_ood_noisy" in summary: add_row("evenodd_baseline_ood_noisy", summary["evenodd_base_ood_noisy"])
    if "evenodd_filt_gen_noisy" in summary: add_row("evenodd_filtered_gen_noisy", summary["evenodd_filt_gen_noisy"])
    if "evenodd_filt_ood_noisy" in summary: add_row("evenodd_filtered_ood_noisy", summary["evenodd_filt_ood_noisy"])


    csv_path = summary_dir / "summary.csv"
    import csv
    from make_plots import make_plots

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "acc_sum", "acc_concepts_both", "rs_freq", "rs_rate_given_correct"])
        w.writeheader()
        w.writerows(rows)
    
        
    print(f"[OK] Wrote: {csv_path}")

    print("\nDONE. Next: (optional) add plot generation + noisy-metrics saving into metrics_filter.npz.")


    make_plots(str(csv_path), str(summary_dir / "plots"))


if __name__ == "__main__":
    main()
