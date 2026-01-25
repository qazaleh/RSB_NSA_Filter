# src/make_mnadd_largesmall.py
"""
Generate MNAdd-LargeSmall dataset from raw MNIST .gz files.

Train split (GEN): both digits in {0..4}
OOD test split: at least one digit in {5..9}

Outputs:
- train.npz
- test_gen.npz   (both digits in {0..4} from MNIST test set)
- test_ood.npz   (at least one digit in {5..9} from MNIST test set)
"""

from __future__ import annotations
import argparse
import gzip
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

# Read MNIST images from IDX .gz into shape [N, 28, 28].
def _read_idx_images_gz(path: Path) -> np.ndarray:

    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)

# Read MNIST labels from IDX .gz into shape [N].
def _read_idx_labels_gz(path: Path) -> np.ndarray:

    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)

# Return indices for small digits (0..4) and large digits (5..9).
def _idx_small_large(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    small = np.where(labels <= 4)[0]
    large = np.where(labels >= 5)[0]
    return small, large


def _make_pairs(
    images: np.ndarray,
    labels: np.ndarray,
    n_pairs: int,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    mode in {"SS","SL","LS","LL"}
    S=small(0..4), L=large(5..9)
    Returns X, y_sum, y_d1, y_d2
    """
    small, large = _idx_small_large(labels)

    def sample(pool: np.ndarray, k: int) -> np.ndarray:
        return rng.choice(pool, size=k, replace=True)

    if mode == "SS":
        i1, i2 = sample(small, n_pairs), sample(small, n_pairs)
    elif mode == "SL":
        i1, i2 = sample(small, n_pairs), sample(large, n_pairs)
    elif mode == "LS":
        i1, i2 = sample(large, n_pairs), sample(small, n_pairs)
    elif mode == "LL":
        i1, i2 = sample(large, n_pairs), sample(large, n_pairs)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    d1 = labels[i1].astype(np.uint8)
    d2 = labels[i2].astype(np.uint8)
    y_sum = (d1 + d2).astype(np.uint8)

    x = np.concatenate([images[i1], images[i2]], axis=2)  # [N, 28, 56]
    x = x[:, None, :, :]  # [N, 1, 28, 56]
    return x, y_sum, d1, d2

# Shuffle arrays with the same permutation.
def _shuffle(rng: np.random.Generator, *arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    
    n = arrays[0].shape[0]
    idx = rng.permutation(n)
    return tuple(a[idx] for a in arrays)

# Load MNIST, create LargeSmall splits, and save .npz files.
def main() -> None:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/mnadd_largesmall")
    ap.add_argument("--train_pairs", type=int, default=120000)
    ap.add_argument("--test_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_images = _read_idx_images_gz(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels_gz(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = _read_idx_images_gz(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx_labels_gz(raw_dir / "t10k-labels-idx1-ubyte.gz")

    # TRAIN (GEN): only SS
    X_tr, y_tr, d1_tr, d2_tr = _make_pairs(train_images, train_labels, args.train_pairs, "SS", rng)
    X_tr, y_tr, d1_tr, d2_tr = _shuffle(rng, X_tr, y_tr, d1_tr, d2_tr)

    # TEST GEN: only SS from test set
    n_gen = args.test_pairs // 2
    X_ge, y_ge, d1_ge, d2_ge = _make_pairs(test_images, test_labels, n_gen, "SS", rng)

    # TEST OOD: mix SL and LS (at least one large digit)
    n_ood = args.test_pairs - n_gen
    n_half = n_ood // 2
    X_sl, y_sl, d1_sl, d2_sl = _make_pairs(test_images, test_labels, n_half, "SL", rng)
    X_ls, y_ls, d1_ls, d2_ls = _make_pairs(test_images, test_labels, n_ood - n_half, "LS", rng)

    X_ood = np.concatenate([X_sl, X_ls], axis=0)
    y_ood = np.concatenate([y_sl, y_ls], axis=0)
    d1_ood = np.concatenate([d1_sl, d1_ls], axis=0)
    d2_ood = np.concatenate([d2_sl, d2_ls], axis=0)

    X_ge, y_ge, d1_ge, d2_ge = _shuffle(rng, X_ge, y_ge, d1_ge, d2_ge)
    X_ood, y_ood, d1_ood, d2_ood = _shuffle(rng, X_ood, y_ood, d1_ood, d2_ood)

    np.savez_compressed(out_dir / "train.npz", X=X_tr, y_sum=y_tr, y_d1=d1_tr, y_d2=d2_tr)
    np.savez_compressed(out_dir / "test_gen.npz", X=X_ge, y_sum=y_ge, y_d1=d1_ge, y_d2=d2_ge)
    np.savez_compressed(out_dir / "test_ood.npz", X=X_ood, y_sum=y_ood, y_d1=d1_ood, y_d2=d2_ood)

    print("Done.")
    print("train:", X_tr.shape, "gen:", X_ge.shape, "ood:", X_ood.shape)


if __name__ == "__main__":
    main()
