# JSAC-Grade Experiment Design
## Semantic Channel Learning (SCL) — Complete Redesign

> **Design philosophy:** Every experiment is anchored to a specific theorem or design claim.
> Every table/figure has a single answerable question. Every claim is statistically supported.
> Total compute is estimated so nothing is left open-ended.

---

## Part I: Gaps in the Current Design That Block JSAC

Before prescribing the new design, here is an honest audit of what is missing at JSAC level:

| Gap | Why it blocks JSAC | Fix |
|-----|--------------------|-----|
| Only CIFAR-10 + MNIST sanity check | JSAC expects multi-dataset generalization evidence | Add CIFAR-100 and TinyImageNet |
| IID partition only | Real edge is heterogeneous; JSAC reviewers always ask | Add Dirichlet non-IID (α=0.5, 0.1) |
| SNR sweep only 3 points {10,20,30} dB | Theorem 1 predicts a curve; 3 points is insufficient to fit or falsify it | Fine sweep 0–30 dB in 5 dB steps (7 points) |
| Theorem 1–3 bounds never numerically calibrated | "We use them for trend interpretation" will not survive JSAC review | Estimate L_s, L_ℓ, L_g empirically; overlay bound on data |
| Semantic capacity C_s never measured | Semantic capacity is the paper's core definition; it is defined but never plotted | Measure I(Z̃;Y) proxy and I(X;Z) proxy via VIB across SNR |
| BCBSA advantage not significant | Proposed method cannot fail to beat baselines in at least one scenario | Full statistical test with n≥10 seeds, pre-registered comparison |
| Only 3 attacks | IPM (Inner Product Manipulation) exists in current code; not in paper | Add IPM; add MIN-MAX attack |
| No split-point ablation | Splitting at different layers is a design choice that needs justification | Ablate split at Layer1, Layer2, Layer3 |
| No channel model ablation | Rayleigh + semantic noise vs. OFDM-like vs. Rician — choice is unjustified | Compare channel models |
| Single architecture | Generalization beyond ResNet-18 unknown | Add MobileNetV2 (edge-realistic) |
| Communication cost not measured end-to-end | C_comm formula is theoretical; wall-clock not reported | Measure actual bytes transmitted and wall-clock overhead |
| P2 n=6 seeds, 30 epochs → underpowered | Cohen's d up to 0.458 with n=6 gives power < 0.30 | n=10 seeds, 100 rounds (P2 redesign) |

---

## Part II: Redesigned Experimental Structure

### Five experiment groups, each tied to one paper claim

```
EG-1: Theorem 1 & 3 Calibration     → Does the SNR-accuracy bound hold quantitatively?
EG-2: Semantic Capacity Measurement  → Is C_s measurable and does it track SNR?
EG-3: Robustness Matrix              → Does BCBSA dominate under Byzantine + channel noise jointly?
EG-4: Generalization                 → Do findings hold across datasets, architectures, and heterogeneity?
EG-5: Ablation & Sensitivity         → Which components drive performance?
```

---

## Part III: System Configuration

### III-A. Datasets

| Dataset | Split | Classes | Purpose |
|---------|-------|---------|---------|
| CIFAR-10 | 50k/10k | 10 | Primary benchmark (all EGs) |
| CIFAR-100 | 50k/10k | 100 | Generalization check (EG-4) |
| TinyImageNet | 100k/10k | 200 | Scale check (EG-4) |
| MNIST | 60k/10k | 10 | Sanity cross-check (EG-4) |

### III-B. Data heterogeneity regimes

```
IID       : uniform random partition across K clients
Dir-0.5   : Dirichlet(α=0.5) — mild heterogeneity
Dir-0.1   : Dirichlet(α=0.1) — severe heterogeneity (long-tail per client)
```

Dirichlet partition:
```python
def dirichlet_partition(dataset, num_clients, alpha, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)
    label_indices = {c: np.where(np.array(dataset.targets) == c)[0]
                     for c in range(num_classes)}
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(label_indices[c])).astype(int)
        proportions[-1] = len(label_indices[c]) - proportions[:-1].sum()
        idx = label_indices[c]
        rng.shuffle(idx)
        splits = np.split(idx, np.cumsum(proportions)[:-1])
        for k, s in enumerate(splits):
            client_indices[k].extend(s.tolist())
    return client_indices
```

### III-C. Network architectures

| Model | Client split | Smash dim d_s | Purpose |
|-------|-------------|---------------|---------|
| ResNet-18 (Layer2→3) | Layers 0–2 | 128×16×16 = 32768 | Primary |
| ResNet-18 (Layer1→2) | Layers 0–1 | 64×32×32 = 65536 | Split-point ablation |
| ResNet-18 (Layer3→4) | Layers 0–3 | 256×8×8 = 16384 | Split-point ablation |
| MobileNetV2 (block 4→5) | Blocks 0–4 | ~32×14×14 ≈ 6272 | Edge-realistic arch |

### III-D. Channel models

| Model | Description | σ²_ε formula |
|-------|-------------|--------------|
| **Rayleigh-Semantic** (primary) | h_k ~ CN(0,1); additive latent noise | α/(1+SNR_k) |
| AWGN-Semantic | No fading; deterministic SNR | α/(1+SNR) |
| Rician-Semantic | h_k ~ Rician(ν=1,σ=1); K-factor=1 | α/(1+SNR_k · κ_Rice) |
| Digital-BPSK | Bit-level pipeline: BPSK mod → AWGN → demod → re-quant | BER-based |
| No-channel | σ²_ε = 0 (ideal link) | 0 |

Primary experiments use Rayleigh-Semantic. Channel-model ablation (EG-5) compares all five.

### III-E. SNR sweep

**Fine sweep (EG-1, EG-2):** {0, 5, 10, 15, 20, 25, 30} dB — 7 points  
**Coarse sweep (EG-3, EG-4):** {5, 15, 25} dB — 3 representative points  
**Fixed point (EG-3 attack matrix):** 15 dB — worst-case representative

Per-client heterogeneity: SNR_k ~ SNR_base + N(0, σ²_SNR), σ_SNR = 3 dB

### III-F. Optimizer and schedule

```python
# All experiments
optimizer   = SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
# Warmup (rounds 1–10): linear scale-up from lr=0.01 to 0.1
# Cosine (rounds 11–T): CosineAnnealingLR(T_max=T-10, eta_min=1e-4)
batch_size  = 64   # per client per round
```

### III-G. Seed and statistical design

```
Primary (EG-1, EG-2, EG-3): n = 5 seeds {0,1,2,3,4}
Comparison claims (EG-3 BCBSA vs baselines): n = 10 seeds {0,...,9}
Statistical test: paired Wilcoxon signed-rank (n=5), paired t-test (n=10)
Significance level: α = 0.05 (Bonferroni-corrected for multiple comparisons)
Effect size: Cohen's d reported alongside every p-value
Power target: 0.80 (requires d ≥ 0.90 at n=10 for paired t)
```

Pre-registration rule: the primary comparison is **BCBSA vs FLAME under weight-poisoning at SNR=15 dB, CIFAR-10, IID, 100 rounds**. This is declared before running.

---

## Part IV: Experiment Group Specifications

---

### EG-1: Theorem 1 & 3 Calibration

**Question:** Do the theoretical bounds correctly predict the direction and approximate magnitude of channel degradation?

**Sub-questions:**
1. Does excess loss scale as (1+SNR)^{-1/2} as Theorem 1 predicts?
2. Does gradient variance scale as (1+SNR)^{-1} as Theorem 3 predicts?
3. Can we estimate L_s, L_ℓ, L_g empirically and evaluate bound tightness?

**Configuration:**
```
Dataset       : CIFAR-10, IID
Architecture  : ResNet-18, split at Layer2→3
Channel       : Rayleigh-Semantic
SNR sweep     : {0, 5, 10, 15, 20, 25, 30} dB  (fine, 7 points)
Attack        : none
Defense       : FedAvg (no aggregation noise)
Rounds        : 100
Seeds         : 5
K clients     : 50, all participate
```

**Measurements per round t, per SNR:**

```python
# Measure 1: Excess loss (empirical Δℓ)
excess_loss_t = loss_with_channel(z_tilde) - loss_without_channel(z_clean)

# Measure 2: Gradient variance ratio
grad_var_channel_t = E[||J_t ε_t||²]   # estimated via MC samples
grad_var_data_t    = E[||ξ_data||²]     # estimated from mini-batch variance

# Measure 3: Distortion D_t = ||z_tilde - z||² / d_z
distortion_t = (z_tilde - z_clean).pow(2).mean().item()

# Measure 4: Semantic fidelity
fidelity_t = 1 - distortion_t / z_clean.pow(2).mean().item()

# Measure 5: Gradient Jacobian norm proxy (diagonal approx)
# ||J_t||_2 ≈ max singular value of ∂g/∂z, estimated via power iteration
jacobian_norm_t = power_iter_jacobian_norm(model, z_clean, loss)
```

**Lipschitz constant estimation:**

```python
# Estimate L_s: max ||f_s(z1) - f_s(z2)|| / ||z1 - z2|| over random pairs
def estimate_Ls(server_model, z_samples, n_pairs=500):
    ratios = []
    for _ in range(n_pairs):
        i, j = np.random.choice(len(z_samples), 2, replace=False)
        diff_out = (server_model(z_samples[i]) - server_model(z_samples[j])).norm()
        diff_in  = (z_samples[i] - z_samples[j]).norm()
        if diff_in > 1e-6:
            ratios.append((diff_out / diff_in).item())
    return max(ratios)

# Estimate L_ℓ: similar, over loss values
# Estimate L_g: ||∇θL(z1) - ∇θL(z2)|| / ||z1 - z2||
```

**Bound overlay:**
Compute theoretical bound `B(SNR) = L_ℓ L_s sqrt(d_z α / (1+SNR))` and plot against measured Δℓ(SNR). Report tightness ratio `B(SNR) / Δℓ(SNR)` at each SNR point.

**Outputs:**
- Fig EG1-A: Excess loss Δℓ vs SNR (dB), measured vs theoretical bound (log scale)
- Fig EG1-B: Gradient noise variance vs SNR, channel term vs data term (shows dominance regime)
- Fig EG1-C: Distortion D_t vs round t for each SNR level (convergence)
- Table EG1: L_s, L_ℓ, L_g estimates; bound tightness at {0,15,30} dB; Γ_T values

---

### EG-2: Semantic Capacity Measurement

**Question:** Is semantic capacity C_s measurable, does it track the theoretical bound, and does SCL's IB objective raise it relative to vanilla SFL?

**Sub-questions:**
1. How does measured I(Z̃;Y) compare to I(Z;Y) across SNR? (Channel-induced semantic loss Δ_ch)
2. Does the representation geometry term (½ log det(I + (1+SNR)/α · Σ_z)) match measured I(Z;Z̃)?
3. Does SCL's joint IB objective achieve higher C_s than FedAvg at matched SNR?

**Configuration:**
```
Dataset       : CIFAR-10, IID
Architecture  : ResNet-18, split at Layer2→3
Channel       : Rayleigh-Semantic
SNR sweep     : {0, 5, 10, 15, 20, 25, 30} dB
Attack        : none
Defenses      : {SC²-SFL (SCL), standard FedAvg, sfl_no_channel}
Rounds        : 100
Seeds         : 5
```

**MI estimation (mini-batch proxies):**

```python
# I(X;Z) proxy: KL divergence from variational encoder
def ib_IXZ(mu, logvar):
    # VAE-style: KL(q(z|x) || N(0,I))
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

# I(Z̃;Y) proxy: log|Y| - H(Y|Z̃)
def ib_IZtildeY(logits, y, num_classes=10):
    ce = F.cross_entropy(logits, y)
    return math.log(num_classes) - ce.item()

# Δ_ch: semantic loss from channel
def channel_semantic_loss(z, z_tilde, server_model, y, num_classes=10):
    IZY     = ib_IZtildeY(server_model(z), y, num_classes)
    IZtildeY = ib_IZtildeY(server_model(z_tilde), y, num_classes)
    return IZY - IZtildeY    # Δ_ch ≥ 0 by DPI

# Theoretical geometry bound (Theorem 2, second term)
def capacity_geometry_bound(Sigma_z, snr_db, alpha=1.0):
    snr = 10 ** (snr_db / 10.0)
    scale = (1 + snr) / alpha
    eigvals = torch.linalg.eigvalsh(Sigma_z)
    return 0.5 * torch.log2(1 + scale * eigvals).sum().item()
```

**Σ_z estimation:** Compute empirical covariance of z_k over one epoch at each SNR.

**Semantic efficiency metric:**
```
η_s(SNR) = I(Z̃;Y) / C(SNR)     where C = B log₂(1+SNR)
```
Plot η_s vs SNR for SCL vs FedAvg. Higher η_s = better use of available channel capacity.

**Outputs:**
- Fig EG2-A: I(Z;Y) and I(Z̃;Y) vs SNR for SCL vs FedAvg (shows Δ_ch and whether SCL narrows it)
- Fig EG2-B: Measured I(Z;Z̃) vs theoretical geometry bound (Theorem 2 tightness)
- Fig EG2-C: η_s (semantic efficiency) vs SNR for SCL vs FedAvg vs no-channel
- Table EG2: Δ_ch at {0,10,20,30} dB; η_s values; bound gap

---

### EG-3: Robustness Matrix (Primary Attack-Defense Experiment)

**Question:** Does BCBSA provide statistically significant robustness gains over competitive defenses under the joint presence of Byzantine attacks AND channel noise?

This is the core JSAC contribution experiment. It must be designed for statistical power.

#### EG-3A: Full attack × defense × SNR matrix

**Configuration:**
```
Dataset       : CIFAR-10, IID
Architecture  : ResNet-18, split at Layer2→3
Channel       : Rayleigh-Semantic, SNR ∈ {5, 15, 25} dB
Malicious     : f/N = 0.30 (15/50 clients)
Rounds        : 100  (warmup+cosine)
Seeds         : 5 per cell
```

**Attack taxonomy (6 attacks):**

| ID | Name | Mechanism | Severity |
|----|------|-----------|---------|
| A0 | None | — | — |
| A1 | Weight poisoning | g_adv = g + δ, δ ~ direction × 5.0 | Standard |
| A2 | Label flipping | y' = (y+1) mod C | Standard |
| A3 | SMASH | z̃_adv = z̃ + ε·‖z̃‖·û, ε=0.3 | Semantic-specific |
| A4 | IPM | g_adv = −γ·ĝ + ζ (inner product manipulation) | Adaptive |
| A5 | Min-Max | maximize loss on test while minimizing detection | Adaptive |

IPM implementation:
```python
def ipm_attack(gradients_honest, gamma=10.0):
    # Fang et al. (2020): reverse direction scaled by γ
    mean_honest = torch.stack(gradients_honest).mean(dim=0)
    return -gamma * mean_honest
```

Min-Max implementation:
```python
def minmax_attack(model, data, global_grad, lam=1.0):
    # Maximize task loss subject to gradient being close to honest mean
    adv_loss = -F.cross_entropy(model(data), targets)
    detection_penalty = lam * (grad_adv - global_grad).norm()
    return adv_loss + detection_penalty
```

**Defense set (7 defenses):**

| ID | Name | Reference |
|----|------|-----------|
| D0 | FedAvg | McMahan et al. 2017 |
| D1 | Coord. Median | Yin et al. 2018 |
| D2 | Krum | Blanchard et al. 2017 |
| D3 | FLAME | Nguyen et al. 2022 |
| D4 | FLTrust | Cao et al. 2022 |
| D5 | DnC | Shejwalkar & Houmansadr 2021 |
| D6 | **BCBSA** | This paper |
| D6-abl | BCBSA-NoSem | Ablation: ω₁=ω₂=0 |

**Matrix dimensions:** 6 attacks × 8 defenses × 3 SNR = **144 cells × 5 seeds = 720 runs**

**Primary statistical test (pre-registered):**
```
H₀: accuracy(BCBSA) ≤ accuracy(FLAME) under A1, SNR=15dB
H₁: accuracy(BCBSA) > accuracy(FLAME) under A1, SNR=15dB
Test: one-sided paired t-test, n=10 seeds
α = 0.05, target power = 0.80
```

Additional Bonferroni-corrected tests for A4 (IPM), A3 (SMASH), A5 (MinMax).

#### EG-3B: Byzantine fraction sweep

**Configuration:**
```
f/N ∈ {0.10, 0.20, 0.30, 0.40, 0.50}
Attack  : A1 (weight poisoning), A4 (IPM)
Defenses: FLAME, BCBSA, Median, FedAvg
SNR     : 15 dB
Seeds   : 5
Rounds  : 100
```

Plot accuracy vs f/N for each defense × attack combination.
Mark the Theorem 5 failure boundary for Median (f/N crossing 0.5).

#### EG-3C: Theorem 5 empirical instance

Design this as a controlled verification:
```
N = 50, f ∈ {10, 15, 20, 25} (f/N = 0.2, 0.3, 0.4, 0.5)
Attack: weight poisoning with uniform bias b ≠ 0 (||b||=1)
Defense: Coord. Median
Measure: ||ĝ - g||₂ for each f/N
Expected: ||ĝ - g||₂ = ||b|| > 0 for all f < N/2
```

This isolates the theorem from the ML accuracy proxy and gives a direct algebraic verification.

#### EG-3D: Runtime-robustness Pareto frontier

Measure wall-clock time per round and final accuracy under A1 for all 8 defenses.
Plot the Pareto frontier: accuracy vs normalized runtime.
BCBSA's target position: better accuracy than FLAME at lower or comparable overhead.

---

### EG-4: Generalization

**Question:** Do the SCL gains (when present) and the BCBSA robustness advantage generalize beyond the primary CIFAR-10 IID setting?

#### EG-4A: Multi-dataset

```
Datasets      : CIFAR-10, CIFAR-100, TinyImageNet, MNIST
Architecture  : ResNet-18 (adapted per dataset)
SNR           : 15 dB (single representative point)
Attack        : A1 (weight poisoning), f/N=0.30
Defense       : FedAvg, FLAME, BCBSA
Rounds        : 100 (CIFAR-10/100), 80 (TinyImageNet), 50 (MNIST)
Seeds         : 3 per cell
```

Report: accuracy table across datasets; defense ranking stability (is FLAME > BCBSA > Median consistent?).

#### EG-4B: Data heterogeneity (non-IID)

```
Partition     : IID, Dir-0.5, Dir-0.1
Dataset       : CIFAR-10
SNR           : 15 dB
Attack        : A1, f/N=0.30
Defense       : FedAvg, FLAME, BCBSA, FedProx
Rounds        : 100
Seeds         : 5
```

Non-IID changes two things simultaneously:
1. Local data drift (affects learning convergence)
2. Gradient heterogeneity (interacts with BCBSA trust scores — more heterogeneous gradients may trigger false positives)

Report: accuracy vs heterogeneity level; BCBSA false-positive rate (honest clients incorrectly excluded).

#### EG-4C: Architecture generalization

```
Models        : ResNet-18, MobileNetV2
Dataset       : CIFAR-10
SNR           : {5, 15, 25} dB
Attack        : none, A1
Defense       : FedAvg, FLAME, BCBSA
Rounds        : 100
Seeds         : 3
```

MobileNetV2 is edge-realistic (designed for on-device inference), making it a more compelling architectural choice for the 6G edge narrative.

---

### EG-5: Ablation and Sensitivity

#### EG-5A: Split-point ablation

```
Split points  : Layer1→2 (d_s=65536), Layer2→3 (d_s=32768), Layer3→4 (d_s=16384)
Dataset       : CIFAR-10
SNR           : {5, 15, 25} dB
Attack        : none, A1
Defense       : FedAvg, BCBSA
Rounds        : 100
Seeds         : 3
```

Three competing effects as the split goes deeper:
- Client compute ↑ (good for privacy, bad for edge device)
- d_s ↓ (less communication overhead)
- Distortion D_k ↓ (smaller d_s means less noise from σ²_ε·d_z)

Report: accuracy, communication cost B_tx, convergence speed per split point.

#### EG-5B: Channel model comparison

```
Channel models: Rayleigh-Semantic, AWGN-Semantic, Rician-Semantic, Digital-BPSK, No-channel
Dataset       : CIFAR-10
SNR           : {5, 15, 25} dB
Attack        : none
Defense       : SC²-SFL (SCL objective)
Rounds        : 100
Seeds         : 3
```

This directly addresses the reviewers' concern about the isotropic noise approximation.
Key question: does the performance ordering change across channel models?

#### EG-5C: BCBSA component ablation

```
Variants:
  V0: BCBSA-Full     (ω₁=1, ω₂=1, ω₃=0.1, temporal=True)
  V1: BCBSA-NoFid    (ω₁=0, ω₂=1, ω₃=0.1)   — remove fidelity F_k
  V2: BCBSA-NoDist   (ω₁=1, ω₂=0, ω₃=0.1)   — remove distortion D_k
  V3: BCBSA-NoSem    (ω₁=0, ω₂=0, ω₃=0.1)   — no semantic signals
  V4: BCBSA-NoTemp   temporal=False            — remove temporal regularization
  V5: FLAME                                    — external baseline for calibration

Dataset       : CIFAR-10, IID
SNR           : 15 dB
Attacks       : A1, A4 (IPM)
f/N           : 0.30
Rounds        : 100
Seeds         : 5
```

This is the definitive ablation. Every BCBSA component gets isolated credit. Without this, JSAC reviewers will reject the BCBSA claim.

#### EG-5D: IB coefficient sensitivity

```
Sweep β ∈ {0.1, 0.5, 1.0, 2.0, 5.0}  (IB relevance weight)
Sweep λ ∈ {1e-3, 5e-3, 1e-2, 5e-2}   (compression regularization)
Fixed: μ=0.01, α=1.0, 15 dB, IID, CIFAR-10, no attack, 3 seeds
```

Report a 5×4 heatmap of final test accuracy. Identify the stable plateau (the region where results are insensitive to ±2× coefficient change).

#### EG-5E: Compression ratio sensitivity

```
ρ_c ∈ {0.3, 0.5, 0.7, 0.85, 0.95}
Metrics: accuracy, B_tx (bits/round), F (fidelity)
Fixed: 15 dB, IID, CIFAR-10, no attack, 5 seeds, 100 rounds
```

This directly tests the communication-accuracy tradeoff that is central to the JSAC audience.

---

## Part V: Metrics (Complete and Operationalized)

Every metric has an exact formula, measurement point, and storage key.

```python
metrics_per_round = {
    # Learning
    "test_accuracy"        : float,   # N_correct / N_total on held-out test set
    "train_loss"           : float,   # mean cross-entropy over participating clients
    "excess_loss"          : float,   # E[ℓ(fs(z̃),y)] - E[ℓ(fs(z),y)]

    # Channel
    "semantic_fidelity"    : float,   # 1 - ||z-z̃||² / ||z||²,  mean over clients
    "distortion_Dk"        : float,   # ||z̃-z||² / d_z, mean over clients
    "snr_effective_db"     : float,   # mean realized SNR (post-fading)
    "ber"                  : float,   # bit error rate (digital baseline only)

    # Information-theoretic
    "IXZ_proxy"            : float,   # KL-based I(X;Z) estimate (bits)
    "IZtildeY_proxy"       : float,   # CE-based I(Z̃;Y) estimate (bits)
    "delta_ch"             : float,   # I(Z;Y) - I(Z̃;Y) (channel semantic loss)
    "semantic_efficiency"  : float,   # I(Z̃;Y) / C(SNR) = η_s

    # Aggregation / robustness
    "acceptance_ratio"     : float,   # |R| / K in BCBSA
    "false_positive_rate"  : float,   # honest clients excluded by defense
    "gradient_bias_norm"   : float,   # ||E[g̃] - g||₂ (Theorem 4 proxy)
    "grad_var_channel"     : float,   # channel noise contribution to grad var
    "grad_var_data"        : float,   # data noise contribution to grad var

    # Communication
    "bytes_transmitted"    : int,     # actual bytes per round (compressed)
    "compression_ratio"    : float,   # ρ_k (adaptive or fixed)
    "wall_clock_sec"       : float,   # wall-clock time per round

    # Lipschitz proxies (EG-1 only)
    "Ls_estimate"          : float,
    "Lg_estimate"          : float,
    "bound_tightness"      : float,   # B(SNR) / measured Δℓ
}
```

---

## Part VI: Statistical Testing Protocol

### Pre-registered primary comparison
```
Test      : BCBSA vs FLAME, A1 (weight poisoning), SNR=15 dB, CIFAR-10 IID
Type      : One-sided paired t-test (H₁: BCBSA > FLAME)
n         : 10 seeds
α         : 0.05
Required  : Report exact p-value, Cohen's d, 95% CI on difference
```

### Secondary comparisons (Bonferroni-corrected, m=4 tests)
```
Adjusted α: 0.05/4 = 0.0125
Tests: BCBSA vs FLAME under {A3-SMASH, A4-IPM, A5-MinMax, non-IID Dir-0.1}
```

### EG-4 generalization (within-dataset paired tests, n=3)
For n=3, minimum achievable exact p = 0.125 (not powerful). Report effect size and CI only; do not claim significance.

### Convergence comparison
Use area under the convergence curve (AUC of accuracy vs round) as a single scalar per run. Compare AUC distributions via Wilcoxon signed-rank when normality is suspect.

---

## Part VII: Figures and Tables (Full Specification)

### Tables

| # | Title | Source | Rows × Cols | Key statistic |
|---|-------|---------|-------------|---------------|
| T1 | System parameter summary | all EGs | — | — |
| T2 | Lipschitz estimates and bound tightness | EG-1 | 7 SNR × 4 metrics | L_s, L_ℓ, L_g, B/Δℓ |
| T3 | Semantic capacity measurement | EG-2 | 7 SNR × 5 metrics | I(Z;Y), I(Z̃;Y), Δ_ch, η_s |
| T4 | Full robustness matrix (primary) | EG-3A | 6 attacks × 8 defenses | mean±std, SNR=15 dB |
| T5 | Multi-dataset generalization | EG-4A | 4 datasets × 3 defenses | accuracy, defense ranking |
| T6 | Non-IID robustness | EG-4B | 3 partitions × 4 defenses | accuracy, false-positive rate |
| T7 | BCBSA component ablation | EG-5C | 6 variants × 2 attacks | accuracy, Δ vs full BCBSA |
| T8 | Coefficient sensitivity | EG-5D | 5β × 4λ heatmap | accuracy |
| T9 | Communication-accuracy tradeoff | EG-5E | 5 ρ_c values | accuracy, B_tx, F |
| T10 | Hypothesis test summary | EG-3 | 5 tests | p-value, d, 95% CI |

### Figures

| # | Title | Source | Type | What it shows |
|---|-------|---------|------|---------------|
| F1 | System architecture | — | Block diagram | SCL pipeline |
| F2 | Excess loss vs SNR: theory vs measured | EG-1 | Line+band | Theorem 1 calibration |
| F3 | Gradient noise decomposition | EG-1 | Stacked area | Channel vs data noise over SNR |
| F4 | I(Z;Y) and I(Z̃;Y) vs SNR | EG-2 | Dual-line | Semantic capacity measurement |
| F5 | Semantic efficiency η_s vs SNR | EG-2 | Line | SCL vs FedAvg channel utilization |
| F6 | Robustness heatmap (attacks × defenses) | EG-3A | Heatmap | Primary matrix, SNR=15 dB |
| F7 | Convergence trajectories | EG-3A | Line | Clean vs adversarial cases |
| F8 | Byzantine fraction sweep | EG-3B | Line | Accuracy vs f/N per defense |
| F9 | Theorem 5 gradient bias verification | EG-3C | Scatter | ||ĝ-g|| vs f/N for Median |
| F10 | Pareto frontier: robustness vs runtime | EG-3D | Scatter | Security-efficiency tradeoff |
| F11 | Non-IID accuracy + false positive rate | EG-4B | Bar+line | Heterogeneity robustness |
| F12 | Split-point ablation | EG-5A | Bar | Accuracy and B_tx per split |
| F13 | BCBSA component contribution | EG-5C | Bar | Variant ablation under A1, A4 |

---

## Part VIII: Compute Estimate

```
EG-1: 7 SNR × 1 defense × 1 attack × 5 seeds × 100 rounds       =   3,500 round-executions
EG-2: 7 SNR × 3 methods × 1 attack × 5 seeds × 100 rounds        =  10,500
EG-3A: 6 attacks × 8 defenses × 3 SNR × 5 seeds × 100 rounds     =  72,000
EG-3B: 5 f/N × 2 attacks × 4 defenses × 5 seeds × 100 rounds     =  20,000
EG-3C: 4 f/N × 1 defense × 1 attack × 5 seeds × 100 rounds       =   2,000
EG-3D: 8 defenses × 1 attack × 5 seeds × 100 rounds               =   4,000
EG-4A: 4 datasets × 3 defenses × 2 attacks × 3 seeds × 100 rounds = 14,400
EG-4B: 3 partitions × 4 defenses × 2 attacks × 5 seeds × 100 rounds = 12,000
EG-4C: 2 architectures × 3 SNR × 2 attacks × 3 seeds × 100 rounds =   3,600
EG-5A: 3 splits × 2 SNR × 2 attacks × 3 seeds × 100 rounds        =   3,600
EG-5B: 5 channels × 3 SNR × 1 attack × 3 seeds × 100 rounds       =   4,500
EG-5C: 6 variants × 2 attacks × 5 seeds × 100 rounds               =   6,000
EG-5D: 20 combos × 3 seeds × 100 rounds                            =   6,000
EG-5E: 5 ρ_c × 5 seeds × 100 rounds                               =   2,500
─────────────────────────────────────────────────────────────────────────────
TOTAL                                                               ≈ 164,600 round-executions

At 50 clients, 64 batch, ResNet-18: ~1.2 sec/round on A100
164,600 × 1.2s ≈ 197,520 sec ≈ 55 GPU-hours (single A100)
With 4 parallel jobs: ~14 GPU-hours wall-clock
```

---

## Part IX: What Changes in the Paper

When experiments are complete, the following paper sections must be updated:

| Section | Current state | Required change |
|---------|--------------|-----------------|
| §VI.A datasets | CIFAR-10 + MNIST | Add CIFAR-100, TinyImageNet; describe Dirichlet partition |
| §VI.B architecture | ResNet-18 only | Add MobileNetV2; justify split point via EG-5A |
| §VI.C channel model | 3-point SNR sweep | 7-point fine sweep; add channel model comparison table |
| §VI.D attacks | 3 attacks | 6 attacks including IPM and Min-Max |
| §VI.E defenses | 6 defenses | 8 defenses including FLTrust, DnC; add Theorem 5 verification subsection |
| §VI.G hyperparameters | Grid search description | Report EG-5D sensitivity heatmap as figure |
| §VII.A | P2 direct comparison | Replace with EG-2 semantic capacity measurement |
| §VII.B | Hypothesis tests with n=6 | Replace with n=10 pre-registered tests, Bonferroni-corrected |
| §VII.C | SNR accuracy (0.54% range) | Replace with fine sweep + Theorem 1 overlay (EG-1) |
| §VII.D | Fidelity scatter | Add η_s semantic efficiency figure (EG-2) |
| §VII.E | Attack robustness | Full 6×8 matrix; Byzantine sweep; Theorem 5 algebraic verification |
| §VII.F | Defense comparison | BCBSA ablation table (EG-5C); Pareto frontier |
| §VII.G | BCBSA ablation (weak) | Full 6-variant ablation with significance tests |
| §VIII discussion | "bounded-response behavior" | Reframe using Theorem 3: channel term vs data term decomposition |
| Markov chain notation | X→Z→Z̃→Y (causal) | Fix to Y−X−Z−Z̃ (undirected Markov) |

---

## Part X: Run Order (Parallelization Plan)

```
Priority 1 (blocking): EG-3A (primary matrix, 720 runs) — start immediately
Priority 2 (for theory): EG-1, EG-2 (bound calibration, capacity measurement)
Priority 3 (for ablation): EG-5C (BCBSA components), EG-5A (split point)
Priority 4 (generalization): EG-4A, EG-4B
Priority 5 (sensitivity): EG-5B, EG-5D, EG-5E, EG-3B, EG-3C, EG-3D, EG-4C

Suggested GPU assignment (4 GPUs):
GPU 0: EG-3A (CIFAR-10, attacks × defenses × SNR)
GPU 1: EG-1 + EG-2 (theory calibration)
GPU 2: EG-5C + EG-5A + EG-3B (ablations + sweep)
GPU 3: EG-4A + EG-4B + EG-4C (generalization)
```

---

## Part XI: Acceptance Criteria

The paper is ready to submit to JSAC when ALL of the following are satisfied:

### Must-have (hard gates)
- [ ] EG-3A complete: BCBSA shows p < 0.05 (one-sided, n=10) vs FLAME under **at least one** attack scenario
- [ ] EG-1 complete: Theorem 1 bound overlaid on measured excess loss at 7 SNR points
- [ ] EG-2 complete: I(Z̃;Y) measured and η_s plotted vs SNR for SCL vs FedAvg
- [ ] EG-5C complete: all 6 BCBSA variants ablated under A1 and A4
- [ ] EG-4B complete: non-IID results included with false-positive rate reported
- [ ] Hypothesis test table: distinct p-values, n reported, Bonferroni correction applied
- [ ] Markov chain notation fixed: Y−X−Z−Z̃

### Should-have (quality gates)
- [ ] EG-4A: at least CIFAR-100 results included (TinyImageNet optional if time-constrained)
- [ ] EG-3C: algebraic Theorem 5 verification (direct gradient bias measurement)
- [ ] EG-5A: split-point ablation with communication cost comparison
- [ ] EG-3D: Pareto frontier figure with wall-clock measurements
- [ ] Algorithm 3 in paper updated to reflect temporal regularization (BCBSA-v2)

### Nice-to-have (reviewer bonus)
- [ ] EG-4C: MobileNetV2 results
- [ ] EG-5B: channel model comparison
- [ ] Full TinyImageNet results
