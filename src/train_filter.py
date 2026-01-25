# src/train_filter.py
"""
Train an NSA-style MLP filter that takes the baseline CNN digit probabilities and outputs
corrected digit probabilities (for d1 and d2), then evaluate RS / GEN / OOD again.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Loads MNAdd .npz files and returns (image, d1_label, d2_label, sum_label).
class MNAddDataset(Dataset):


    def __init__(self, npz_path: str):
        # Load arrays saved by the dataset generator
        data = np.load(npz_path)
        self.X = data["X"]            # uint8, shape [N, 1, 28, 56]
        self.y_d1 = data["y_d1"]      # uint8, shape [N]
        self.y_d2 = data["y_d2"]      # uint8, shape [N]
        self.y_sum = data["y_sum"]    # uint8, shape [N]

    def __len__(self) -> int:
        # Return dataset size
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert one sample to torch tensors
        x = torch.from_numpy(self.X[idx]).float() / 255.0
        d1 = torch.tensor(int(self.y_d1[idx]), dtype=torch.long)
        d2 = torch.tensor(int(self.y_d2[idx]), dtype=torch.long)
        s = torch.tensor(int(self.y_sum[idx]), dtype=torch.long)
        return x, d1, d2, s

# Baseline CNN encoder with two classifier heads (digit1 and digit2).
class BaselineCNN(nn.Module):
    

    def __init__(self):
        super().__init__()
        # Feature extractor for 1x28x56 images
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Two heads for predicting digits
        self.head_d1 = nn.Linear(128, 10)
        self.head_d2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode image then output logits for digit1 and digit2
        h = self.features(x).flatten(1)
        return self.head_d1(h), self.head_d2(h)

# Small MLP that takes 20 probs (10 for d1 + 10 for d2) and outputs corrected logits.
class ConceptFilterMLP(nn.Module):
   

    def __init__(self, hidden: int = 64):
        super().__init__()
        # Simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(20, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 20),
        )

    def forward(self, probs20: torch.Tensor) -> torch.Tensor:
        # Output corrected logits (shape [B,20]) for (d1,d2)
        return self.net(probs20)

# Compute baseline digit probabilities and return a concatenated 20-dim prob vector.
def _probs_from_baseline(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
   
    logits_d1, logits_d2 = model(x)
    p1 = torch.softmax(logits_d1, dim=1)
    p2 = torch.softmax(logits_d2, dim=1)
    return torch.cat([p1, p2], dim=1)  # [B,20]

@torch.no_grad()
def evaluate_with_filter(
    baseline: nn.Module,
    filt: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    noise_sigma: float = 0.0,
) -> dict:
    """
    Evaluate baseline + optional filter on a dataset split.

    If noise_sigma > 0, we add Gaussian noise to the input images at evaluation time
    (x is assumed to be normalized in [0,1]).
    """
    baseline.eval()
    if filt is not None:
        filt.eval()

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

        # Add small Gaussian noise to input images (x in [0,1]). (evaluation-only)
        if noise_sigma > 0.0:
            noise = torch.randn_like(x) * noise_sigma
            x = torch.clamp(x + noise, 0.0, 1.0)

        # Get baseline digit probabilities
        probs20 = _probs_from_baseline(baseline, x)  # [B, 20]

        if filt is not None:
            corr_logits = filt(probs20)              # [B, 20]
            corr_probs = torch.softmax(corr_logits, dim=1)
            p1 = corr_probs[:, :10]
            p2 = corr_probs[:, 10:]
        else:
            p1 = probs20[:, :10]
            p2 = probs20[:, 10:]

        # Predict digits and sum (symbolic reasoning step)
        pred_d1 = p1.argmax(dim=1)
        pred_d2 = p2.argmax(dim=1)
        pred_sum = pred_d1 + pred_d2

        total += x.size(0)

        c1 = (pred_d1 == d1)
        c2 = (pred_d2 == d2)
        both = c1 & c2

        d1_correct += int(c1.sum().item())
        d2_correct += int(c2.sum().item())
        both_digits_correct += int(both.sum().item())

        task_ok = (pred_sum == y_sum)
        sum_correct += int(task_ok.sum().item())

        # Reasoning shortcut (RS): correct sum but wrong digit concept(s)
        rs = task_ok & (~both)
        rs_count += int(rs.sum().item())

    return {
        "n": total,
        "acc_d1": d1_correct / total,
        "acc_d2": d2_correct / total,
        "acc_concepts_both": both_digits_correct / total,
        "acc_sum": sum_correct / total,
        "rs_freq": rs_count / total,
        "rs_rate_given_correct": (rs_count / sum_correct) if sum_correct > 0 else 0.0,
        "noise_sigma": noise_sigma,
    }

# Train the concept-filter MLP on top of a frozen baseline CNN, then evaluate.
def main() -> None:
    
    # Noisy eval (example sigma)
    sigma = 0.10

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/mnadd_evenodd")
    ap.add_argument("--baseline_ckpt", type=str, default="results/baseline_cnn.pt")
    ap.add_argument("--out_dir", type=str, default="results_filter")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train_npz = data_dir / "train.npz"
    gen_npz = data_dir / "test_gen.npz"
    ood_npz = data_dir / "test_ood.npz"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets/loaders
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

    # Load and freeze baseline
    baseline = BaselineCNN().to(device)
    baseline.load_state_dict(torch.load(args.baseline_ckpt, map_location=device))
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad_(False)

    # Create filter MLP
    filt = ConceptFilterMLP(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(filt.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    # Evaluate baseline before training filter
    base_gen = evaluate_with_filter(baseline, None, gen_loader, device)
    base_ood = evaluate_with_filter(baseline, None, ood_loader, device)
    print("BASELINE | GEN:", base_gen)
    print("BASELINE | OOD:", base_ood)

    # Train filter to correct digit predictions
    for epoch in range(1, args.epochs + 1):
        filt.train()
        running_loss = 0.0
        n_seen = 0

        for x, d1, d2, _ in train_loader:
            x = x.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)

            with torch.no_grad():
                probs20 = _probs_from_baseline(baseline, x)  # frozen baseline output

            opt.zero_grad()
            corr_logits20 = filt(probs20)  # [B,20]

            # Split into two 10-class problems
            logits1 = corr_logits20[:, :10]
            logits2 = corr_logits20[:, 10:]
            loss = ce(logits1, d1) + ce(logits2, d2)

            loss.backward()
            opt.step()

            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        avg_loss = running_loss / max(1, n_seen)

        # Evaluate filtered model each epoch
        filt_gen = evaluate_with_filter(baseline, filt, gen_loader, device)
        filt_ood = evaluate_with_filter(baseline, filt, ood_loader, device)

        print(f"Epoch {epoch:02d} | filter_loss {avg_loss:.4f} | "
              f"GEN sum {filt_gen['acc_sum']:.4f} RS {filt_gen['rs_freq']:.4f} | "
              f"OOD sum {filt_ood['acc_sum']:.4f} RS {filt_ood['rs_freq']:.4f}")

    # Save filter checkpoint
    ckpt_path = out_dir / "filter_mlp.pt"
    torch.save(filt.state_dict(), ckpt_path)
    print(f"Saved filter model to: {ckpt_path}")

    # Final metrics (baseline vs filtered)(noisy vs nonise)

    base_gen_noisy = evaluate_with_filter(baseline, None, gen_loader, device, noise_sigma=sigma)
    base_ood_noisy = evaluate_with_filter(baseline, None, ood_loader, device, noise_sigma=sigma)

    filt_gen_noisy = evaluate_with_filter(baseline, filt, gen_loader, device, noise_sigma=sigma)
    filt_ood_noisy = evaluate_with_filter(baseline, filt, ood_loader, device, noise_sigma=sigma)


    np.savez(
        out_dir / "metrics_filter.npz",
        base_gen_noisy=base_gen,
        base_ood_noisy=base_ood,
        filt_gen_noisy=filt_gen,
        filt_ood_noisy=filt_ood,
    )
    print(f"BASELINE NOISY sigma={sigma} | GEN:", base_gen_noisy)
    print(f"BASELINE NOISY sigma={sigma} | OOD:", base_ood_noisy)
    print(f"FILTERED NOISY sigma={sigma} | GEN:", filt_gen_noisy)
    print(f"FILTERED NOISY sigma={sigma} | OOD:", filt_ood_noisy)   


if __name__ == "__main__":
    main()
