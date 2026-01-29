# RSB_NSA_Filter

This repository contains the code for an AI course project studying **reasoning shortcuts (RS)** in simple neuro-symbolic models.  
The project is inspired by **RSBench** and **Neural-Symbolic Abduction (NSA)**, and focuses on a lightweight, practical setting.

The main objective is to analyze **when reasoning shortcuts occur**, how they are affected by ambiguity (noise), and whether a small neural filtering module can influence RS behavior.

---

## Project Overview

The project implements a neural–symbolic arithmetic reasoning pipeline:

1. **Perception stage**  
   A convolutional neural network predicts symbolic concepts (digits) from MNIST-based images.

2. **Symbolic reasoning stage**  
   A fixed symbolic rule computes the final arithmetic result from predicted concepts.

3. **Evaluation**  
   The final answer is compared with internal concept predictions to detect reasoning shortcuts.

A reasoning shortcut occurs when the final symbolic answer is correct, but one or more predicted concepts are incorrect.

---

## Datasets

Two synthetic arithmetic datasets are generated from MNIST:

- **MNAdd-EvenOdd**  
  Arithmetic reasoning with parity-based concepts (even / odd).

- **MNAdd-LargeSmall**  
  Arithmetic reasoning with magnitude-based concepts (large / small).

Datasets are generated automatically and are **not stored** in the repository.

---

## Neural Filtering (NSA-inspired)

Inspired by Neural-Symbolic Abduction (NSA), the project introduces a **small neural filtering module** placed between perception and symbolic reasoning.

The filter learns to reweight predicted concepts before reasoning, with the goal of studying how this affects:
- task accuracy
- reasoning shortcut frequency
- robustness under noise and out-of-distribution (OOD) settings

This is a simplified adaptation, not a full NSA system.

---

## Environment

- Python: 3.10
- Framework: PyTorch (CPU-only)
- Environment manager: Conda
  
---

## Recreate environment

```bash
conda env create -f environment.yml
conda activate rsb-nsa
```
---

## How to run
The main entry point is main.py.
Quick run (recommended)
Uses existing trained models and results, regenerates summaries and plots:

```bash
python main.py --skip_train
```
---
This command:
- checks raw data 
- regenerates result tables 
- regenerate plots
- prints an organized summary in the terminal
  
---

## Full Reproduction (Traning from scratch)
To retrain all models and overwrite existing results:
```bash
python main.py --force --epochs_baseline 5 --epochs_filter 10
```
You may change the number of epochs if desired.

---

## Results
All final results are saved in:
```bash
results_summary/
```
This folder contains:
- `summary.json`
- `summary.csv`
- `plots/`(figures used in the report)

---

Results include:
- clean (in-distribution) evaluation
- out-of-distribution (OOD) evaluation
- noisy evaluation for reasoning shortcut analysis

---

## Repository Structure
```text
.
├── main.py
├── make_plots.py
├── src/
│   ├── make_mnadd_evenodd.py
│   ├── make_mnadd_largesmall.py
│   ├── train_baseline.py
│   └── train_filter.py
├── results_summary/
├── environment.yml
├── requirements.txt
└── README.md
```
---

## Notes
- MNIST is downloaded automatically when required.
- All experiments are designed to run on CPU.
- The project prioritizes clarity, reproducibility, and analysis over performance optimization.





