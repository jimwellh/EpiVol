# EpiVol

**Contagious Semantics: A Mass-Conserving SIRS State-Space Filter for News-Driven Volatility**

_Do epidemic state-space filters improve news-driven volatility forecasts?_

> **Status:** Work in progress. No results have been reported yet.

---

## Research Question

When converting irregular financial news embeddings into daily features, does a **causal state-space filter with SIR epidemic dynamics** reduce 1-to-10-day realized volatility forecast errors compared to zero-filling, forward-filling, or monotonic exponential decay?

## Motivation

Daily realized volatility lives on a regular trading-day grid, while news arrives as sparse, irregular events with multi-day silent periods ("information blackouts"). Any forecasting pipeline must choose a causal filter that turns the history of sparse text shocks into a daily latent state. Common choices each embed a strong assumption:

| Filter                   | Implicit assumption                          | Failure mode                               |
| ------------------------ | -------------------------------------------- | ------------------------------------------ |
| Impulse / zero-fill      | Market digests news instantly                | Ignores multi-day underreaction and drift  |
| Forward-fill (LOCF)      | Old news stays fully fresh                   | Injects stale signal during long blackouts |
| Exponential decay (EWMA) | Peak impact at arrival, then geometric decay | Cannot represent delayed peaks             |

Information diffusion research (Hong & Stein, 1999; Shive, 2010; Meursault et al., 2023) suggests news is digested gradually through investor contagion. A discrete-time **SIRS** filter yields a **hump-shaped memory kernel**: the state can amplify on silent days while the susceptible pool is large (`S_{t-1} > γ/β`), peak with a delay, and then decay — while remaining mass-conserving (`S + I + R = 1`) and bounded.

## Methods

All text filters receive the same daily input `m̃_t · u_t ∈ R^19` and output a dimension-matched state `z_t ∈ R^19`, which feeds an identical HAR-RV + Ridge readout:

```
ŷ_{t+h} = μ_i + θ₁ v_t^(d) + θ₂ v_t^(w) + θ₃ v_t^(m) + w_hᵀ z_t,    h ∈ {1, 2, 3, 5, 10}
```

| Model          | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| Baseline 0     | Text-free HAR-RV                                                                |
| Baseline 1     | Impulse / zero-fill                                                             |
| Baseline 2     | Causal forward-fill (LOCF)                                                      |
| Baseline 3     | Monotonic exponential decay, causal AR(1)                                       |
| **Proposed 1** | Mass-conserving causal SIRS state-space filter (split contagion / spark update) |
| **Proposed 2** | Second-order causal Gamma filter (linear hump-shaped fallback)                  |
| **Proposed 3** | Context-dependent learnable SIRS filter (PyTorch, BPTT)                         |
| Ablation       | `R^19` vs. `R^20` state (appending susceptible capacity `S_t`)                  |

## Data

- **Source:** [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) — financial news aligned with daily OHLCV ([Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID)).
- **Universe:** 50 S&P 500 stocks, stratified by training-period news coverage into high / medium / sparse tiers (`seed = 42`).
- **Text features:** frozen [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert) sentiment probabilities (3-D) + training-only PCA of `[CLS]` embeddings (16-D).
- **Target:** log Parkinson (1980) range-based realized volatility.
- **Alignment:** 16:00:00 America/New_York forecast origin; after-hours, weekend, and holiday news routes to the next NYSE trading day.

| Split      | Period                  | Use                                             |
| ---------- | ----------------------- | ----------------------------------------------- |
| Train      | 2016-01-01 – 2020-12-31 | Fit scalers, PCA, coverage tiers, Ridge weights |
| Validation | 2021-01-01 – 2022-12-31 | Hyperparameter selection                        |
| Test       | 2023-01-01 – 2023-12-31 | Held out until final evaluation                 |

## Repository Structure

```
EpiVol/
├── configs/            # Seeds, date splits, horizons, grid-search ranges
├── data/
│   ├── raw/            # Raw FNSPID files (not tracked)
│   ├── processed/      # Aligned daily panels and FinBERT features (not tracked)
│   └── metadata/       # Stratified ticker list and coverage tiers
├── docs/
│   ├── blueprints/     # Project blueprint and experiment notes
│   ├── bibliography/   # Annotated literature review
│   └── milestones/     # Course deliverables (proposal → final paper)
├── notebooks/          # Exploratory analysis and diagnostics
├── src/
│   ├── data_pipeline/  # FNSPID filtering, timestamp alignment, ticker stratification
│   ├── features/       # FinBERT extraction, PCA, realized-volatility targets
│   ├── filters/        # Baseline filters, SIRS filter, learnable SIRS
│   ├── evaluation/     # Ridge forecaster, block bootstrap, blackout stress test
│   └── plotting/       # Paper figures
└── tests/              # Leakage and causality tests
```

## Setup

Requires Python 3.11+ and, for FinBERT inference, a CUDA-capable GPU (CPU works but is slow).

```bash
git clone https://github.com/jimwellh/EpiVol.git
cd EpiVol
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download FNSPID into `data/raw/` before running the pipeline. Run the test suite with:

```bash
pytest -q
```

Pipeline entry points and usage will be documented as modules are implemented.

## Roadmap

- [x] Milestone 1 — Project proposal
- [ ] Milestone 2 — Baseline checkpoint (data pipeline, leakage tests, baseline validation results)
- [ ] Milestone 3 — Early paper draft (SIRS and Gamma filters, impulse-response figure)
- [ ] Milestone 4 — Analysis complete (test set, bootstrap CIs, ablations, stress test)
- [ ] Milestone 5 — Paper and video draft
- [ ] Milestone 6 — Final NeurIPS-style paper and five-minute video

## Author

Ching-Wei Huang — University of Pennsylvania
