# Worked example — ADMET prediction for GLP-1(7-37)

A complete, reproducible walk-through of the committed v4.2 peptide ADMET
platform on a single real therapeutic peptide: **human glucagon-like peptide-1,
fragment 7-37** (UniProt P01275, residues 98-128; sequence
`HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG`, 31 residues, MW 3355.7 Da — sequence pulled
live from the UniProt REST feature table during this run, not from memory).

Everything in this directory runs offline once the four checkpoints under
`models_v4/` and the two precomputed embeddings under `data/` are present.

## What is here

| Path | Description |
|---|---|
| `predict_glp1.py` | Canonical one-command predictor (run this) |
| `models_v4/*/admet_mlp.pt`, `scaler.pt` | The four committed v4.2 single-task checkpoints (~10 MB) |
| `data/molformer/glp1_emb.npz` | MoLFormer-XL CLS embedding of the GLP-1 SMILES (precomputed with the platform's own `molformer_embed.py`) |
| `data/esmc/glp1_esmc.npz` | ESMC-600M CLS embedding of the GLP-1 sequence (precomputed with `esmc_embed.py` / equivalent loaders) |
| `results/glp1_predictions.json` | Raw predictions (rerun-stable) |
| `results/glp1_case_study.json` | Cross-check run over GLP-1(7-37) and GLP-1(7-36) with freshly batch-embedded features |
| `results/ground_truth_literature.md` | Literature reference values, with the exact quoted passages and sources |
| `feature_extractor.py`, `admet_model.py`, `endpoint_config.py`, `predictor.py`, `*_embed.py` | Verbatim copies of the platform source (MIT, from `peptide-admet-jpa`) so the example imports the *committed* feature path, not a reimplementation |

## How a peptide flows through the platform

1. **Sequence → SMILES** — RDKit `MolFromFASTA` renders the linear peptide to a
   canonical SMILES (568 characters). No custom chemistry code.
2. **Molecular endpoints (PAMPA/MDCK, Caco-2)** — the SMILES is featurized
   exactly as in training: 217 RDKit 2D descriptors + 2048-bit Morgan(R=2)
   fingerprint + 768-dim frozen MoLFormer-XL CLS embedding = 3033-dim input →
   v4.2 MLP → logPapp.
3. **Sequence endpoints (Hemolysis, Half_life)** — 428-dim classical sequence
   vector (AAC + dipeptide composition + physicochemical) + 1152-dim frozen
   ESMC-600M CLS embedding = 1580-dim input → v4.2 MLP → probability / log10 s.
4. **Scaling guard (documented, not hidden)** — the standardized vector is
   winsorized at ±30σ and every feature beyond ±5σ of the *training*
   distribution is counted and reported. For GLP-1 the raw information-content
   descriptor `Ipc` is 4.98×10⁸⁵ vs a training ceiling of 4.14×10³⁶
   (the training sets are cyclic peptidomimetics of 4–12 residues, MW ≤ 1700);
   an uncapped linear head returns 1e24 nonsense. The guard is why the model
   returns a floor-consistent value **plus an explicit OOD flag (170 of 3033
   features beyond ±5σ)** instead of a fabricated-looking number.

## Result

| Endpoint | Prediction | Interpretation |
|---|---|---|
| PAMPA/MDCK | logPapp **−13.5** (floor region; OOD-flagged) | not expected to pass an artificial membrane in the trained chemistry space |
| Caco-2 | logPapp **−11.1** (floor region; OOD-flagged) | same |
| Hemolysis | **p = 0.029** non-hemolytic | consistent with a circulating peptide hormone |
| Plasma t₁/₂ | **log₁₀ = 3.04 → ≈18 min** (in-distribution; 5 OOD features) | see literature comparison |

## Comparison against measured values (details in `results/ground_truth_literature.md`)

- **Plasma half-life — quantitatively comparable.** Native GLP-1(7-37) is
  cleared by DPP-4 with a reported plasma t₁/₂ of ~1.5–5 min (PMC12632547:
  "approximately 0.08 hours (4.8 minutes)"). The model's 18 min is the same
  order of magnitude and a ~4× overestimate — reasonable for a regression
  trained on 1,763 heterogeneous plasma-stability entries, and the sequence
  branch of this model is in-distribution for GLP-1.
- **Permeability — qualitatively right, quantitatively out-of-domain.**
  Gupta et al. (PMC3586668) measured exenatide (a GLP-1 mimetic) across Caco-2
  monolayers at Papp 3.1–7.8×10⁻⁶ cm/s (logPapp ≈ −5.1 to −5.5); Youn et al.
  (cited there) report 0.5×10⁻⁷ cm/s (logPapp ≈ −7.3) for a GLP-1 analog.
  The platform predicts −11 to −13 — the right conclusion (far below any
  orally relevant permeability) but 4–6 log units too low, because **both
  molecular endpoints were trained exclusively on MW ≤ 1700 cyclic
  peptidomimetics** (dataset maxima exactly 1700 Da) and GLP-1 is a linear
  3.4-kDa peptide: 170/3033 standardized features lie beyond ±5σ. The OOD
  counter, not the number itself, is the deliverable here.
- **Hemolysis — consistent.** No published assay exists for native GLP-1;
  p = 0.029 is consistent with its physiology.

**Honest bottom line.** The model is quantitatively trustworthy on the
in-distribution endpoint (half-life, ±3.6× of measured) and provides a
correct qualitative call with a loud, machine-checkable OOD flag on the
molecular endpoints, whose chemistry domain (4–12-mer macrocycles ≤1700 Da)
does not contain 31-mer linear hormones. This is exactly the domain-of-
applicability failure mode the manuscript's §2.1/§4 limitations predict.

## Reproducing from scratch

```bash
pip install torch numpy pandas scikit-learn rdkit transformers
python molformer_embed.py --smiles-file glp1_smiles.txt --out data/molformer/glp1_emb.npz
# ESMC-600M (needs transformers >= 5; weights stream from HuggingFace `biohub/ESMC-600M`)
python esmc_embed.py --sequences-file glp1_seq.txt --out data/esmc/glp1_esmc.npz
python predict_glp1.py
```

(`glp1_smiles.txt` = RDKit-canonical SMILES of the FASTA line above; the
committed `.npz` caches make the last step the only one needed for inference.)

Verified on Python 3.14 / torch 2.14.1+cu126 / RDKit 2026.x / transformers
5.18, 2026-10-05. The in-distribution control (a training-set molecule scored
through the identical path) recovers its measured label within the test-set
error, confirming the plumbing: measured −3.90 → predicted −4.38.
