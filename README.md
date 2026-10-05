# newadmet

Revised manuscript (v2.1) for the peptide ADMET benchmark study
*Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation*,
revised in response to expert peer review of `peptide-admet-jpa`, **with the
review-closing experiments actually executed and committed**.

## Contents
| Path | Description |
|---|---|
| `peptide_admet_manuscript_jcim_v2.md` | Revised main manuscript (v2.1) |
| `peptide_admet_manuscript_jcim_blind_v2.md` | Anonymized copy (automated leak-check clean) |
| `REVISION_NOTES.md` | Point-by-point reviewer response / change log |
| `analysis/ceiling_convention.py` | Reproducible oracle-ceiling convention (reproduces 0.5387 / 0.5696 exactly) |
| `analysis/bootstrap_ci.py` | Paired bootstrap 95% CI for ΔR² (10k resamples, common random numbers) |
| `analysis/dump_predictions.py` | Per-molecule prediction dump schema for all routes |
| `analysis/em_reconstruction.py` | EM censored-label reconstruction probe (§4.3) |
| `analysis/run_multisplit.py`, `retrain_splits.py`, `run_tabpfn_*.py` | §3.7 multi-split replication incl. split-matched baseline re-training |
| `results/predictions/*.npz` | Committed per-molecule test predictions (21 files) |
| `results/ci_summary.json`, `results/multisplit_final.json` | Executed CI tables |
| `figures/fig1–3.png` | Rendered figures (scatter w/ floor, decomposition, ceiling attainment) |

## Headline results (v2.1)
- PAMPA: baseline 0.464 → TabPFN v2 **0.496** (bootstrap CI +0.033 [+0.009, +0.055], P(Δ≤0)=0.003) → KPGT fine-tune **0.513**
- Caco-2: baseline 0.391 (re-measured from checkpoint) → TabPFN v2 **0.442** (CI +0.053 [+0.020, +0.088])
- **Multi-split replication (§3.7)**: TabPFN beats split-matched re-trained baselines on 6/6 split×endpoint pairs, all CIs exclude zero
- Oracle ceilings (declared convention, `analysis/ceiling_convention.py`): 0.5387 / 0.5696
- Limitation 4 (single split) closed for the TabPFN route; open for KPGT (per-molecule dumps pending)

## Status / remaining work before resubmission
1. Re-dump KPGT test predictions (`--dump-test-predictions`) to attach bootstrap CIs to the KPGT deltas.
2. Zenodo DOI for the frozen archive (promised in Data Availability).
3. KPGT multi-split replication (GPU-days).
4. Graphical abstract render.

MIT License (carried over from `peptide-admet-jpa`).
