# src/train_baseline.py
"""
Train a baseline CNN to predict the two MNIST digits (d1, d2) from a concatenated 56x28 image.
Then we can compute the sum symbolically as d1_pred + d2_pred.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Loads MNAdd-EvenOdd .npz files and returns (image, d1_label, d2_label, sum_label).
class MNAddDataset(Dataset):
    
    # Load arrays saved by make_mnadd_evenodd.py
    def __init__(self, npz_path: str):
       
        data = np.load(npz_path)
        self.X = data["X"]            # uint8, shape [N, 1, 28, 56]
        self.y_d1 = data["y_d1"]      # uint8, shape [N]
        self.y_d2 = data["y_d2"]      # uint8, shape [N]
        self.y_sum = data["y_sum"]    # uint8, shape [N]

    # Return dataset size
    def __len__(self) -> int:    
        return int(self.X.shape[0])

    # Convert one sample to torch tensors

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        x = torch.from_numpy(self.X[idx]).float() / 255.0  # normalize to [0,1]
        d1 = torch.tensor(int(self.y_d1[idx]), dtype=torch.long)
        d2 = torch.tensor(int(self.y_d2[idx]), dtype=torch.long)
        s = torch.tensor(int(self.y_sum[idx]), dtype=torch.long)
        return x, d1, d2, s

# Small CNN encoder with two classifier heads: one for digit1 and one for digit2.
class BaselineCNN(nn.Module):
    

    def __init__(self):
        super().__init__()
        # Feature extractor for 1x28x56 images
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> [32, 28, 56]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [32, 14, 28]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [64, 14, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [64, 7, 14]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> [128, 7, 14]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # -> [128, 1, 1]
        )
        # Two heads for predicting digits (0..9)
        self.head_d1 = nn.Linear(128, 10)
        self.head_d2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode image and output logits for digit1 and digit2
        h = self.features(x).flatten(1)  # [B, 128]
        logits_d1 = self.head_d1(h)
        logits_d2 = self.head_d2(h)
        return logits_d1, logits_d2

# Evaluate digit accuracy, sum accuracy, and RS frequency on a dataset split.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    
    model.eval()

    total = 0
    d1_correct = 0
    d2_correct = 0
    both_digits_correct = 0
    sum_correct = 0
    rs_count = 0

    for x, d1, d2, y_sum in loader:
        x = x.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        y_sum = y_sum.to(device)

        logits_d1, logits_d2 = model(x)
        pred_d1 = logits_d1.argmax(dim=1)
        pred_d2 = logits_d2.argmax(dim=1)

        pred_sum = pred_d1 + pred_d2

        total += x.size(0)

        c1 = (pred_d1 == d1)
        c2 = (pred_d2 == d2)

        d1_correct += int(c1.sum().item())
        d2_correct += int(c2.sum().item())
        both = c1 & c2
        both_digits_correct += int(both.sum().item())

        task_ok = (pred_sum == y_sum)
        sum_correct += int(task_ok.sum().item())

        # RS: sum is correct but at least one digit concept is wrong
        rs = task_ok & (~both)
        rs_count += int(rs.sum().item())

    return {
        "n": total,
        "acc_d1": d1_correct / total,
        "acc_d2": d2_correct / total,
        "acc_concepts_both": both_digits_correct / total,  # CONL-style
        "acc_sum": sum_correct / total,
        "rs_freq": rs_count / total,
        "rs_rate_given_correct": (rs_count / sum_correct) if sum_correct > 0 else 0.0,
    }

# Train baseline CNN on MNAdd-EvenOdd train split and evaluate on GEN and OOD test splits.
def main() -> None:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/mnadd_evenodd")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)  # Mac CPU safe
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train_npz = data_dir / "train.npz"
    gen_npz = data_dir / "test_gen.npz"
    ood_npz = data_dir / "test_ood.npz"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets/loaders
    train_ds = MNAddDataset(str(train_npz))
    gen_ds = MNAddDataset(str(gen_npz))
    ood_ds = MNAddDataset(str(ood_npz))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)

    device = torch.device("cpu")
    model = BaselineCNN().to(device)

    # Training setup
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    # Train loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for x, d1, d2, _ in train_loader:
            x = x.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            opt.zero_grad()
            logits_d1, logits_d2 = model(x)
            loss = ce(logits_d1, d1) + ce(logits_d2, d2)
            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        avg_loss = running_loss / max(1, n_seen)

        # Quick eval each epoch
        gen_metrics = evaluate(model, gen_loader, device)
        ood_metrics = evaluate(model, ood_loader, device)

        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | "
              f"GEN sum {gen_metrics['acc_sum']:.4f} RS {gen_metrics['rs_freq']:.4f} | "
              f"OOD sum {ood_metrics['acc_sum']:.4f} RS {ood_metrics['rs_freq']:.4f}")

    # Save checkpoint
    ckpt_path = out_dir / "baseline_cnn.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved baseline model to: {ckpt_path}")

    # Final eval + save metrics
    gen_metrics = evaluate(model, gen_loader, device)
    ood_metrics = evaluate(model, ood_loader, device)

    metrics_path = out_dir / "baseline_metrics.npz"
    np.savez(metrics_path, gen=gen_metrics, ood=ood_metrics)
    print(f"Saved metrics to: {metrics_path}")
    print("GEN metrics:", gen_metrics)
    print("OOD metrics:", ood_metrics)


if __name__ == "__main__":
    main()
