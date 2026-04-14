# SCL Experiment — Semantic Channel Learning

> **JSAC-grade experiment suite for Byzantine-robust Split Federated Learning over semantic wireless channels.**

This repository contains the complete implementation of the five experiment groups described in [`jsac_experiment_design.md`](jsac_experiment_design.md), covering theorem calibration, semantic capacity measurement, robustness evaluation, generalization, and ablation studies.

---

## Table of Contents

1. [Background](#background)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Experiment Groups](#experiment-groups)
   - [EG-1: Theorem Calibration](#eg-1-theorem-1--3-calibration)
   - [EG-2: Semantic Capacity](#eg-2-semantic-capacity-measurement)
   - [EG-3: Robustness Matrix](#eg-3-robustness-matrix)
   - [EG-4: Generalization](#eg-4-generalization)
   - [EG-5: Ablation & Sensitivity](#eg-5-ablation--sensitivity)
7. [Components](#components)
   - [Datasets & Partitioning](#datasets--partitioning)
   - [Model Architectures](#model-architectures)
   - [Channel Models](#channel-models)
   - [Byzantine Attacks](#byzantine-attacks)
   - [Aggregation Defenses](#aggregation-defenses)
   - [Metrics](#metrics)
8. [Outputs](#outputs)
9. [Running Tests](#running-tests)
10. [Citation](#citation)
11. [Changelog](#changelog)
12. [License](#license)

---

## Background

Split Federated Learning (SFL) distributes a neural network across edge clients and a central server, transmitting intermediate feature representations (*smash data*) over a wireless channel. This introduces two concurrent threats:

| Threat | Description |
|--------|-------------|
| **Channel noise** | Rayleigh/AWGN fading degrades smash-data fidelity as a function of SNR |
| **Byzantine clients** | Malicious participants inject poisoned gradients or corrupted updates |

**BCBSA** (*Byzantine-Channel-aware Blind Semantics-Aware Aggregation*) — the proposed defense — computes a trust score for each client using semantic fidelity, channel distortion, and gradient cosine similarity, then performs weighted aggregation. The experiments validate the theoretical bounds (Theorem 1 & 3) and benchmark BCBSA against six baselines across five attack types.

---

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python | 3.8 |
| PyTorch | 2.0.0 |
| torchvision | 0.15.0 |
| NumPy | 1.24.0 |
| SciPy | 1.10.0 |
| pandas | 2.0.0 |
| matplotlib | 3.7.0 |
| seaborn | 0.12.0 |
| scikit-learn | 1.2.0 |
| tqdm | 4.65.0 |

---

## Installation

```bash
git clone https://github.com/ispandey/scl-experiment.git
cd scl-experiment

# (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                 # installs the scl package in editable mode
```

Datasets (CIFAR-10, CIFAR-100, TinyImageNet, MNIST) are downloaded automatically by torchvision on the first run. When no network is available a **synthetic fallback dataset** is used automatically so the pipeline can always be validated offline.

---

## Quick Start

### Validate the pipeline (no GPU, no data download needed)

```bash
python runner.py --exp eg1 --device cpu --dry_run
```

The dry-run uses 4 clients, 2 rounds, 1 seed, and 2 SNR points with synthetic data — it completes in under 2 minutes on a laptop CPU.

### Run a single experiment group

```bash
# Full EG-1: Theorem calibration (7 SNR points × 5 seeds × 100 rounds)
python runner.py --exp eg1 --device cuda --output_dir results/

# Full EG-3A: 6 attacks × 8 defenses × 3 SNR (the robustness matrix)
python runner.py --exp eg3a --device cuda --output_dir results/

# All five experiment groups sequentially
python runner.py --exp all --device cuda --output_dir results/
```

### CLI reference

```
python runner.py --help

  --exp          {eg1,eg2,eg3a,eg3b,eg3c,eg3d,eg3,eg4a,eg4b,eg4c,eg4,
                  eg5a,eg5b,eg5c,eg5d,eg5e,eg5,all}
  --output_dir   Root directory for results  [default: results]
  --device       cpu or cuda                 [default: cpu]
  --dry_run      Minimal run to validate the pipeline
  --num_rounds   Override number of rounds
  --num_seeds    Override number of random seeds
```

---

## Project Structure

```
scl-experiment/
├── runner.py                    # CLI entry point
├── setup.py
├── requirements.txt
├── jsac_experiment_design.md    # Full experiment specification
│
├── scl/                         # Main Python package
│   ├── config.py                # EG1Config – EG5Config dataclasses
│   ├── data/
│   │   ├── datasets.py          # CIFAR-10/100, TinyImageNet, MNIST, SyntheticDataset
│   │   └── partition.py         # IID and Dirichlet(α) non-IID partitioning
│   ├── models/
│   │   ├── resnet.py            # Split ResNet-18 (split points 1/2/3)
│   │   └── mobilenet.py         # Split MobileNetV2
│   ├── channels/
│   │   ├── rayleigh.py          # Rayleigh-Semantic (primary)
│   │   ├── awgn.py              # AWGN-Semantic
│   │   ├── rician.py            # Rician-Semantic
│   │   ├── digital.py           # Digital-BPSK baseline
│   │   └── noiseless.py         # Noiseless (upper bound)
│   ├── attacks/
│   │   ├── weight_poison.py     # A1: Weight Poisoning
│   │   ├── label_flip.py        # A2: Label Flipping
│   │   ├── smash.py             # A3: SMASH (feature-space attack)
│   │   ├── ipm.py               # A4: Inner Product Manipulation
│   │   └── minmax.py            # A5: Min-Max
│   ├── defenses/
│   │   ├── fedavg.py            # D0: FedAvg (baseline)
│   │   ├── median.py            # D1: Coordinate-wise Median
│   │   ├── krum.py              # D2: Krum
│   │   ├── flame.py             # D3: FLAME
│   │   ├── fltrust.py           # D4: FLTrust
│   │   ├── dnc.py               # D5: DnC (Divide-and-Conquer)
│   │   └── bcbsa.py             # D6: BCBSA (proposed) + ablation variants
│   ├── metrics/
│   │   ├── information.py       # MI proxies: I(X;Z), I(Z̃;Y), Δ_ch, η_s
│   │   ├── lipschitz.py         # L_s, L_ℓ, L_g estimators + Theorem 1 bound
│   │   ├── robustness.py        # Backdoor ASR, clean-label accuracy
│   │   ├── communication.py     # Bytes transmitted, wall-clock overhead
│   │   └── aggregated.py        # RoundMetrics dataclass, MetricsTracker
│   ├── training/
│   │   ├── client.py            # SFLClient
│   │   ├── server.py            # SFLServer
│   │   ├── federated.py         # FederatedTrainer (main training loop)
│   │   └── scheduler.py         # WarmupCosineScheduler
│   ├── experiments/
│   │   ├── _utils.py            # build_trainer, set_seed, save_tracker
│   │   ├── eg1_theorem_calibration.py
│   │   ├── eg2_semantic_capacity.py
│   │   ├── eg3_robustness_matrix.py
│   │   ├── eg4_generalization.py
│   │   └── eg5_ablation.py
│   └── analysis/
│       ├── stats.py             # Paired t-test, Wilcoxon, Bonferroni, Cohen's d
│       ├── figures.py           # 13 publication figures (F2–F13)
│       └── tables.py            # 10 LaTeX/CSV tables (T2–T10)
│
├── tests/                       # 58 unit + integration tests
│   ├── test_channels.py
│   ├── test_attacks.py
│   ├── test_defenses.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_integration.py
│
└── results/                     # Auto-created; one sub-folder per experiment
    ├── eg1/
    ├── eg2/
    └── ...
```

---

## Experiment Groups

### EG-1: Theorem 1 & 3 Calibration

**Research question:** Does the SNR–accuracy bound from Theorem 1 hold quantitatively?

| Hyperparameter | Value |
|----------------|-------|
| Dataset | CIFAR-10, IID |
| Architecture | ResNet-18, split layer 2 |
| Channel | Rayleigh-Semantic |
| SNR sweep | 0, 5, 10, 15, 20, 25, 30 dB |
| Defense | FedAvg (no attacks) |
| Seeds × Rounds | 5 × 100 |

**Outputs:** `results/eg1/`

- `seed{s}_snr{db}.json` — per-seed-snr round history
- `summary.csv` — aggregated over seeds
- Figures F2 (excess loss vs SNR), F3 (bound tightness)
- Table T2 (Lipschitz estimates)

```bash
python runner.py --exp eg1 --device cuda
```

---

### EG-2: Semantic Capacity Measurement

**Research question:** Is the semantic capacity *C*_s measurable and does it track SNR?

Compares SCL, FedAvg, and a non-channel baseline. Measures mutual information proxies I(Z̃;Y) and I(X;Z) via the VIB approximation.

```bash
python runner.py --exp eg2 --device cuda
```

**Outputs:** `results/eg2/` — Figures F4 (capacity vs SNR), F5 (MI proxies), Table T3.

---

### EG-3: Robustness Matrix

**Research question:** Does BCBSA dominate under Byzantine attacks and channel noise jointly?

Four sub-experiments:

| Sub-exp | Description |
|---------|-------------|
| **EG-3A** | Full 6 attacks × 8 defenses × 3 SNR matrix |
| **EG-3B** | Malicious-fraction sweep (10–50 %) |
| **EG-3C** | Channel model comparison (Rayleigh / AWGN / Rician / Digital-BPSK) |
| **EG-3D** | Statistical significance (paired t-test, Bonferroni-corrected, n=10 seeds) |

```bash
python runner.py --exp eg3  --device cuda   # all four sub-experiments
python runner.py --exp eg3a --device cuda   # matrix only
```

**Attacks:** none · weight_poison · label_flip · smash · ipm · minmax  
**Defenses:** fedavg · median · krum · flame · fltrust · dnc · bcbsa · bcbsa_nosem

**Outputs:** `results/eg3{a,b,c,d}/` — Figures F6–F9, Tables T4–T6.

---

### EG-4: Generalization

**Research question:** Do findings hold across datasets, architectures, and data heterogeneity?

| Sub-exp | Variable |
|---------|----------|
| **EG-4A** | Four datasets: CIFAR-10, CIFAR-100, TinyImageNet, MNIST |
| **EG-4B** | Three heterogeneity regimes: IID, Dir-0.5, Dir-0.1 |
| **EG-4C** | Two architectures: ResNet-18, MobileNetV2 |

```bash
python runner.py --exp eg4 --device cuda
```

**Outputs:** `results/eg4{a,b,c}/` — Figures F10–F11, Tables T7–T8.

---

### EG-5: Ablation & Sensitivity

**Research question:** Which design choices drive BCBSA's performance?

| Sub-exp | Ablated variable |
|---------|-----------------|
| **EG-5A** | Split layer (1 / 2 / 3) |
| **EG-5B** | Channel model (5 variants) |
| **EG-5C** | BCBSA component ablation (6 variants incl. no-fidelity, no-distortion, no-temporal) |
| **EG-5D** | Hyperparameter sensitivity (ω₁,ω₂,ω₃,β, λ) |
| **EG-5E** | Compression ratio sweep |

```bash
python runner.py --exp eg5 --device cuda
```

**Outputs:** `results/eg5{a,b,c,d,e}/` — Figures F12–F13, Table T9–T10.

---

## Components

### Datasets & Partitioning

```python
from scl.data.datasets import get_dataset
from scl.data.partition import iid_partition, dirichlet_partition

ds = get_dataset("cifar10", train=True)        # auto-downloads; synthetic fallback offline
splits_iid = iid_partition(ds, num_clients=50, seed=0)
splits_dir = dirichlet_partition(ds, num_clients=50, alpha=0.1, seed=0)
```

### Model Architectures

| Model | Split point | Smash shape (CIFAR 32×32) |
|-------|-------------|--------------------------|
| ResNet-18 | layer 1 | (B, 64, 8, 8) |
| ResNet-18 | layer 2 (default) | (B, 128, 4, 4) |
| ResNet-18 | layer 3 | (B, 256, 2, 2) |
| MobileNetV2 | block boundary | (B, C, H, W) |

```python
from scl.models.resnet import build_resnet18_split
client_model, server_model = build_resnet18_split(split_layer=2, num_classes=10)
```

### Channel Models

| Name | Key | Description |
|------|-----|-------------|
| Rayleigh-Semantic | `rayleigh` | Primary; fading amplitude + semantic noise |
| AWGN-Semantic | `awgn` | Additive Gaussian only |
| Rician-Semantic | `rician` | LoS component + scatter |
| Digital-BPSK | `digital` | Hard-decision quantise → modulate → demodulate |
| Noiseless | `noiseless` | Identity (upper bound) |

```python
from scl.channels import get_channel
ch = get_channel("rayleigh")
z_tilde = ch(z, snr_db=15.0)
```

### Byzantine Attacks

| ID | Name | Key |
|----|------|-----|
| A1 | Weight Poisoning | `weight_poison` |
| A2 | Label Flipping | `label_flip` |
| A3 | SMASH (feature-space) | `smash` |
| A4 | Inner Product Manipulation | `ipm` |
| A5 | Min-Max | `minmax` |

```python
from scl.attacks import get_attack
attack = get_attack("minmax")
```

### Aggregation Defenses

| ID | Name | Key |
|----|------|-----|
| D0 | FedAvg | `fedavg` |
| D1 | Coordinate-wise Median | `median` |
| D2 | Krum | `krum` |
| D3 | FLAME | `flame` |
| D4 | FLTrust | `fltrust` |
| D5 | DnC | `dnc` |
| D6 | **BCBSA** (proposed) | `bcbsa` |
| — | BCBSA ablations | `bcbsa_nofid`, `bcbsa_nodist`, `bcbsa_nosem`, `bcbsa_notemp` |

```python
from scl.defenses import get_defense
defense = get_defense("bcbsa")
```

### Metrics

```python
from scl.metrics.lipschitz import estimate_Ls, compute_theorem1_bound
from scl.metrics.information import ib_IZtildeY, semantic_efficiency
from scl.metrics.aggregated import MetricsTracker
```

Each training round produces a `RoundMetrics` object that captures:

- `test_accuracy`, `train_loss`, `excess_loss`
- `semantic_fidelity`, `channel_distortion`
- `Ls_estimate`, `Lg_estimate`, `bound_tightness`
- `bytes_transmitted`, `wall_clock_sec`

---

## Outputs

Every experiment writes its results to `results/<exp>/`:

| File | Description |
|------|-------------|
| `seed{s}_snr{db}.json` | Full round-by-round history for one (seed, SNR) run |
| `summary.csv` | Aggregated DataFrame over all seeds and conditions |
| `*.tex` / `*.csv` | Publication-ready tables (LaTeX + CSV) |
| `figures/F*.pdf` | Publication-ready figures at 150 dpi |

---

## Running Tests

```bash
# All 58 tests
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_defenses.py -v
python -m pytest tests/test_integration.py -v
```

Tests cover: channels, attacks, defenses, models, data loading, metrics, and end-to-end federated training rounds. All tests use a synthetic dataset so no internet connection is required.

---

## Citation

If you use this code or experimental design in your work, please cite:

```bibtex
@article{pandey2025scl,
  title   = {Semantic Channel Learning: Byzantine-Robust Split Federated Learning
             over Semantic Wireless Channels},
  author  = {Shaurya, Satyam},
  journal = {IEEE Journal on Selected Areas in Communications},
  year    = {2025},
  note    = {Code: \url{https://github.com/ispandey/scl-experiment}}
}
```

---

## Changelog

### 2026-04-14

- **`scl/metrics/information.py` — semantic efficiency unit fix.**  
  `semantic_efficiency()` now correctly converts I(Z̃;Y) from **nats to bits** (÷ ln 2) before dividing by the Shannon capacity *C* (bits). The previous implementation compared nats directly to bits, producing an η_s value inflated by a factor of ~1.44 (= ln 2⁻¹ ≈ 1.4427).

- **`scl/models/resnet.py` — smash-shape docstring correction.**  
  `ResNet18Client` docstring spatial dimensions were wrong for 32×32 CIFAR input (listed as 64×32×32 / 128×16×16 / 256×8×8). Corrected to match actual output of the torchvision 7×7 stride-2 stem + MaxPool2d: **64×8×8 / 128×4×4 / 256×2×2**.

---

## License

This project is licensed under the MIT License — copyright © 2025 Satyam Shaurya — see the [LICENSE](LICENSE) file for details.
