#!/usr/bin/env python3
"""
Generate MNAdd-EvenOdd dataset from raw MNIST .gz files.

Train split: pairs (even, even) and (odd, odd)
OOD test split: pairs (even, odd) and (odd, even)

Outputs .npz files with:
- X: uint8 images shape [N, 1, 28, 56] (concatenated digits)
- y_sum: uint8 sums in [0..18]
- y_d1: uint8 first digit label in [0..9]
- y_d2: uint8 second digit label in [0..9]
"""

from __future__ import annotations
import argparse
import gzip
import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np 

# Reads MNIST image data from a .gz IDX file and returns it as a NumPy array.
def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)

# Reads MNIST label data from a .gz IDX file and returns it as a NumPy array.
def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)

# Splits digit indices into even and odd groups based on labels.
def _pair_indices(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evens = np.where((labels % 2) == 0)[0]
    odds = np.where((labels % 2) == 1)[0]
    return evens, odds

# Creates MNAdd digit pairs (EE, OO, EO, OE), concatenates images, and generates labels and concepts.
def _make_pairs(
    images: np.ndarray,
    labels: np.ndarray,
    n_pairs: int,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    mode in {"EE", "OO", "EO", "OE"}
    Returns X, y_sum, y_d1, y_d2
    """
    evens, odds = _pair_indices(labels)

# Shuffles multiple arrays using the same random permutation to keep alignment.
    def sample_from(pool: np.ndarray, k: int) -> np.ndarray:
        return rng.choice(pool, size=k, replace=True)

    if mode == "EE":
        i1 = sample_from(evens, n_pairs)
        i2 = sample_from(evens, n_pairs)
    elif mode == "OO":
        i1 = sample_from(odds, n_pairs)
        i2 = sample_from(odds, n_pairs)
    elif mode == "EO":
        i1 = sample_from(evens, n_pairs)
        i2 = sample_from(odds, n_pairs)
    elif mode == "OE":
        i1 = sample_from(odds, n_pairs)
        i2 = sample_from(evens, n_pairs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    d1 = labels[i1].astype(np.uint8)
    d2 = labels[i2].astype(np.uint8)
    y_sum = (d1 + d2).astype(np.uint8)

    # Concatenate horizontally: [28,28] + [28,28] -> [28,56]
    left = images[i1]
    right = images[i2]
    x = np.concatenate([left, right], axis=2)  # (N, 28, 56)
    x = x[:, None, :, :]  # (N, 1, 28, 56)

    return x, y_sum, d1, d2

def _shuffle_in_unison(rng: np.random.Generator, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = arrays[0].shape[0]
    idx = rng.permutation(n)
    return tuple(a[idx] for a in arrays)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/RawData", help="Folder with MNIST .gz files")
    ap.add_argument("--out_dir", type=str, default="data/mnadd_evenodd", help="Output folder")
    ap.add_argument("--train_pairs", type=int, default=120000, help="Total training pairs (EE+OO)")
    ap.add_argument("--test_pairs", type=int, default=20000, help="Total test pairs (EE+OO+EO+OE)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Required MNIST files (your downloaded ones)
    train_images_p = raw_dir / "train-images-idx3-ubyte.gz"
    train_labels_p = raw_dir / "train-labels-idx1-ubyte.gz"
    test_images_p = raw_dir / "t10k-images-idx3-ubyte.gz"
    test_labels_p = raw_dir / "t10k-labels-idx1-ubyte.gz"

    for p in [train_images_p, train_labels_p, test_images_p, test_labels_p]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    train_images = _read_idx_images_gz(train_images_p)
    train_labels = _read_idx_labels_gz(train_labels_p)
    test_images = _read_idx_images_gz(test_images_p)
    test_labels = _read_idx_labels_gz(test_labels_p)

    # Build TRAIN: only EE and OO (EvenOdd training restriction)
    n_train_each = args.train_pairs // 2
    X_ee, y_ee, d1_ee, d2_ee = _make_pairs(train_images, train_labels, n_train_each, "EE", rng)
    X_oo, y_oo, d1_oo, d2_oo = _make_pairs(train_images, train_labels, n_train_each, "OO", rng)

    X_train = np.concatenate([X_ee, X_oo], axis=0)
    y_sum_train = np.concatenate([y_ee, y_oo], axis=0)
    y_d1_train = np.concatenate([d1_ee, d1_oo], axis=0)
    y_d2_train = np.concatenate([d2_ee, d2_oo], axis=0)

    X_train, y_sum_train, y_d1_train, y_d2_train = _shuffle_in_unison(
        rng, X_train, y_sum_train, y_d1_train, y_d2_train
    )

    # Build TEST:
    # We'll create a balanced test with equal parts EE, OO, EO, OE
    n_test_quarter = args.test_pairs // 4
    X_te_ee, y_te_ee, d1_te_ee, d2_te_ee = _make_pairs(test_images, test_labels, n_test_quarter, "EE", rng)
    X_te_oo, y_te_oo, d1_te_oo, d2_te_oo = _make_pairs(test_images, test_labels, n_test_quarter, "OO", rng)
    X_te_eo, y_te_eo, d1_te_eo, d2_te_eo = _make_pairs(test_images, test_labels, n_test_quarter, "EO", rng)
    X_te_oe, y_te_oe, d1_te_oe, d2_te_oe = _make_pairs(test_images, test_labels, n_test_quarter, "OE", rng)

    # GEN test (in-distribution): EE + OO
    X_gen = np.concatenate([X_te_ee, X_te_oo], axis=0)
    y_sum_gen = np.concatenate([y_te_ee, y_te_oo], axis=0)
    y_d1_gen = np.concatenate([d1_te_ee, d1_te_oo], axis=0)
    y_d2_gen = np.concatenate([d2_te_ee, d2_te_oo], axis=0)
    X_gen, y_sum_gen, y_d1_gen, y_d2_gen = _shuffle_in_unison(rng, X_gen, y_sum_gen, y_d1_gen, y_d2_gen)

    # OOD test: EO + OE
    X_ood = np.concatenate([X_te_eo, X_te_oe], axis=0)
    y_sum_ood = np.concatenate([y_te_eo, y_te_oe], axis=0)
    y_d1_ood = np.concatenate([d1_te_eo, d1_te_oe], axis=0)
    y_d2_ood = np.concatenate([d2_te_eo, d2_te_oe], axis=0)
    X_ood, y_sum_ood, y_d1_ood, y_d2_ood = _shuffle_in_unison(rng, X_ood, y_sum_ood, y_d1_ood, y_d2_ood)

    # Save
    np.savez_compressed(
        out_dir / "train.npz",
        X=X_train,
        y_sum=y_sum_train,
        y_d1=y_d1_train,
        y_d2=y_d2_train,
    )
    np.savez_compressed(
        out_dir / "test_gen.npz",
        X=X_gen,
        y_sum=y_sum_gen,
        y_d1=y_d1_gen,
        y_d2=y_d2_gen,
    )
    np.savez_compressed(
        out_dir / "test_ood.npz",
        X=X_ood,
        y_sum=y_sum_ood,
        y_d1=y_d1_ood,
        y_d2=y_d2_ood,
    )

    print("Done.")
    print(f"Saved: {out_dir / 'train.npz'}")
    print(f"Saved: {out_dir / 'test_gen.npz'}")
    print(f"Saved: {out_dir / 'test_ood.npz'}")
    print("Shapes:")
    print("  train X:", X_train.shape, "y_sum:", y_sum_train.shape)
    print("  gen   X:", X_gen.shape, "y_sum:", y_sum_gen.shape)
    print("  ood   X:", X_ood.shape, "y_sum:", y_sum_ood.shape)


if __name__ == "__main__":
    main()