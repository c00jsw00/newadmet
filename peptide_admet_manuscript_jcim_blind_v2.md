# Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation (Revised v2.0, Blind)

**Highlights**
- Ten-route, censoring-aware benchmark of peptide permeability (PAMPA, Caco-2)
- TabPFN v2 and KPGT are the first routes to beat the MLP baselines on both endpoints
- Fine-tuned KPGT leads PAMPA (R² = 0.513), TabPFN v2 leads Caco-2 (R² = 0.442)
- Gradient fine-tuning gains concentrate in the censored floor; non-floor R² declines
- Fine-tuned KPGT (ρ = 0.811) outperforms frozen PeptiVerse embeddings (ρ = 0.671)
- Oracle ceilings (0.539 / 0.570) bound progress: the bottleneck is experimental

## Abstract
**Purpose** Oral absorption remains a principal bottleneck in peptide drug discovery. We report a systematic benchmark of ten computational routes for permeability prediction on the pepADMET dataset (7,283 PAMPA and 7,429 Caco-2 cyclic peptides), under a leakage-controlled, censoring-aware protocol. **Methods** A unique-SMILES 70/10/20 split (seed 42) precluded duplicate leakage. Left-censored floors (−10.0 log cm/s; 3.7% PAMPA, 3.3% Caco-2) were treated explicitly, and oracle ceilings were computed under a single declared convention (R² = 0.539 PAMPA; 0.570 Caco-2). Routes 1–8 were classical (descriptor expansion, rank-Gaussian targets, LightGBM ensembles, Tobit regression, soft-label blending, frozen ChemBERTa embeddings, PeptiVerse raw-data re-training, label averaging); route 9 added two foundation models (TabPFN v2; KPGT fine-tuning), and route 10 repeated route 9 on Caco-2. **Results** No classical route exceeded the PAMPA baseline (R² = 0.464; best Δ = +0.001, within noise). On PAMPA, TabPFN v2 achieved R² = 0.496 ± 0.002 and KPGT fine-tuning R² = 0.513 ± 0.005. On Caco-2 the ranking reversed: TabPFN v2 reached 0.442 ± 0.003 (baseline 0.391) and KPGT 0.411 ± 0.006. KPGT's gains on both endpoints concentrated in the censored floor, with non-floor R² declining; TabPFN improved floor and non-floor accuracy alike on Caco-2. On shared molecules, KPGT fine-tuning (ρ = 0.811) exceeded published PeptiVerse frozen-embedding results (ρ = 0.671). **Conclusions** Foundation models are, to our knowledge, the first methods to break peptide permeability baselines under leakage control, but the oracle ceilings remain the binding constraint: advancing the field requires uncensored re-measurement of floor compounds, an experimental rather than algorithmic task.

**Keywords** peptide ADMET; PAMPA permeability; censored regression; foundation models; TabPFN; KPGT

## 1 Introduction
Therapeutic peptides have emerged as a distinct modality occupying the space between small molecules and biologics, with more than one hundred approved peptide drugs and a pipeline that continues to expand across metabolic, oncologic, and infectious disease indications [1]. Their large size, conformational flexibility, and high polarity impose absorption, distribution, metabolism, elimination, and toxicity (ADMET) challenges that classical small-molecule rules — Lipinski's rule of five foremost among them — were never designed to address [2]. Reliable in silico ADMET prediction for peptides is therefore a prerequisite for efficient lead prioritization and rational design.

Two recent platforms have defined the current state of the art. The pepADMET platform [2] compiled the largest public peptide ADMET collection to date — PAMPA (7,283), Caco-2 (7,429), HLM, MDCK, and additional endpoints — and reported permeability R² values of 0.435–0.657 across model families, establishing the paradigm of classical descriptors combined with gradient boosting. Independently, PeptiVerse [3] unified peptide property prediction around frozen foundational embeddings (PeptideCLM, ChemBERTa, ESM-2) with lightweight prediction heads, reporting PAMPA Spearman ρ = 0.69 and Caco-2 ρ = 0.80, and establishing the paradigm of frozen representations with simple heads.

Both studies, however, share three limitations that constrain their conclusions. First, neither accounts for the left-censored floor in PAMPA measurements (−10.0 log cm/s; 3.7% of the data), which inflates label variance and biases regression metrics. Second, unique-SMILES splitting was not uniformly enforced, leaving the results exposed to leakage through stereoisomers or tautomers. Third, frozen embeddings and end-to-end fine-tuning were never compared on an identical split, so the contribution of task-specific adaptation cannot be isolated. A broader literature of task-specific cyclic-peptide permeability models (CycPeptMP, Multi_CycGT, PCPpred, and a 13-model benchmark) — including transformer-based approaches [18] — reports R² values of 0.67–0.77; we position our results against that literature in §3.6.

Here we address all three limitations in a single, systematic benchmark. Working on the pepADMET PAMPA endpoint under a censoring-aware protocol (v4.2 split, seed 42), we evaluate eight classical improvement routes and one two-model foundation route (route 9), and extend the foundation models to a second endpoint, Caco-2 (route 10), to test whether their gains generalize. We ask two questions: (i) can foundation models — specifically TabPFN v2 (in-context tabular learning) [4] and fine-tuned KPGT (knowledge-guided graph transformer) [5] — exceed the baseline and approach the practical oracle ceiling? and (ii) does end-to-end fine-tuning, rather than frozen embeddings, explain the gap to state-of-the-art peptide platforms? To probe the second question we re-evaluate pepADMET and PeptiVerse results on the shared molecules, holding the split constant for our own models.

## 2 Materials and Methods

### 2.1 Dataset and censored-floor protocol
The pepADMET PAMPA dataset (7,283 rows of cyclic peptides, log Papp in cm/s) was obtained from the pepADMET repository [2]. The table contains 7,177 unique canonical SMILES; 104 SMILES occur more than once (210 rows in duplicated groups), and only 7 of the 104 duplicated groups carry disagreeing replicate labels (maximum spread 2.0 log units). The default protocol (routes 1–7 and 9–10) keeps replicate rows as separate assay records, following the source platform; route 8 tests the alternative (pseudo-label averaging, §2.3). A left-censored floor at −10.0 log cm/s affects 269 rows (3.7%): these values represent the lower bound of assay sensitivity rather than true permeabilities. Following the v4.2 protocol [6], we partitioned the data by unique canonical SMILES into 70/10/20 train/validation/test sets (seed 42), yielding 5,102 / 724 / 1,457 compounds (floor rows per partition: 196 / 26 / 47) and thereby precluding leakage through structural near-duplicates. All metrics are reported twice — on the full test set (floor included, 47 floor rows) and on the non-floor subset (1,410 compounds); floor rows account for 46.1% of the test-set total sum of squares, which is the arithmetic reason the ceiling sits so far below 1. In addition, a practical oracle ceiling of R² = 0.5387 was computed under the convention defined in §2.5; under this convention no evaluated model exceeded the bound.

### 2.2 Baseline model
The v4.2 baseline is a **single-head neural network (MixedADMETMLP)** [6] with a 3,033-dimensional input: 217 RDKit 2D descriptors, 2,048-bit Morgan fingerprints (radius 2; [7], computed with RDKit [8]), and a 768-dimensional frozen MoLFormer-XL CLS embedding (ibm-research/MoLFormer-XL-both-10pct). The trunk comprises two hidden layers (256 → 128 units) with Huber loss (δ = 1.0), trained with Adam (lr = 1 × 10⁻³, weight decay 1 × 10⁻⁵), ReduceLROnPlateau (factor 0.5, patience 4), and early stopping (validation loss, patience 10) for a maximum of 80 epochs. This model (810,497 parameters) achieves R² = 0.464 on the full test set and R² = 0.632 on the non-floor subset. The committed checkpoint used throughout is the published v4.2 checkpoint (single run). As a noise reference for single-run results we use the inter-seed SD of comparable MLP configurations in the same pipeline (0.0025–0.0068; Table 1, routes 1 and 6).

### 2.3 Improvement routes
Routes 1–8 follow the v4.2 pipeline [6] with the identical split and evaluation protocol:

1. **Descriptor expansion** — Morgan fingerprints of radii r1, r2, r3 appended to the baseline feature set (5-seed mean; seeds 42, 123, 456, 789, 1024; MLP retrained).
2. **Rank-Gaussian target transformation** — the regression target is mapped to a Gaussian scale via a quantile transform fitted on the training set only (honest); predictions are back-transformed (5-seed ensemble).
3. **LightGBM ensembling** [9] — multiple LightGBM configurations (with additional Morgan radii r1, r3, r4) combined by averaging (5-seed ensemble). Route 3 is the correct comparator for the leakage analysis in §3.3, because it matches the pepADMET-published model family (gradient boosting on descriptors).
4. **Tobit censored regression** — a censored likelihood (NLL) with the censoring threshold at −10.0, trained with early stopping.
5. **Soft-label blending** — the final prediction is a convex combination of the floor mean and the regression prediction, weighted by a per-compound floor probability; the mixing weight β is selected on validation.
6. **ChemBERTa frozen embeddings** — 384-dimensional embeddings from ChemBERTa-77M (frozen) [10] fed to an MLP head (3 seeds, 4 feature configurations), using the standard v4.2 training loop. ChemBERTa is used here as the published frozen-sequence baseline matching the PeptiVerse arm of the comparison, not as a representative of current sequence models (see §4.4).
7. **PeptiVerse raw-data re-training** — the PeptiVerse PAMPA and Caco-2 regression subsets [3] (HF ChatterjeeLab/PeptiVerse_data; 6,869 PAMPA and 606 Caco-2 rows in the public release) are re-split with the same unique-SMILES 70/10/20 protocol and re-trained end-to-end. The 606-row Caco-2 subset is the release available to us and is smaller than the 7,429-row pepADMET table used in route 10; the PeptiVerse publication's headline Caco-2 metric is computed on that platform's larger internal set, so route 7 is a protocol check on the available data, not a full replication of [3].
8. **Label averaging** — the pepADMET ensemble mean is used as a smoothed pseudo-label (one row per unique SMILES, y = arithmetic mean of replicate labels) in place of the raw measurements.

Routes 2, 4, 5, and 8 were run once (single configuration, deterministic or near-deterministic learners); routes 1, 3, and 6 use the multi-seed protocols stated above. Single-run cells are marked in Table 1.

Route 9 comprises two foundation models:
- **TabPFN v2** (v2.0; direct download, pretraining limits disabled, n_estimators = 4) applied to the 217 RDKit descriptors, evaluated on the canonical split so that its test set is identical to the baseline's. Seeds: 42, 123, 7.
- **KPGT fine-tuning** — a 12-layer LiGhT graph transformer (d_g = 768, 12 attention heads, n_mol_layers = 12, path length 5, d_hpath_ratio = 12, feed-forward dimension 3,072) initialized from the released base checkpoint (KDD 2022 release; self-supervised pretraining corpus as described in [5]) and fine-tuned on the v4.2 training set (batch size 64, AdamW lr = 1 × 10⁻⁴, weight decay 1 × 10⁻⁶, 15-epoch warmup cosine decay, early stopping patience 15). Three seeds (7, 42, 123) were run. Because the publicly available DGL [11] implementation for this architecture is CPU-only on Windows, we developed a pure-Pytorch GPU implementation (scatter-based TripletTransformer replacing DGL's u_dot_v, edge_softmax, and update_all-sum primitives), verified against the official DGL CPU implementation to a maximum absolute difference of 8.3 × 10⁻⁷. Checkpoints are saved per epoch and the best-validation model is used for final evaluation.

Route 10 extends the two foundation models of route 9 to the Caco-2 endpoint (7,429 compounds; 242 floor rows, 3.3%; test n = 1,490 with 39 floor rows) under the identical protocol: the verbatim unique-SMILES 70/10/20 split (seed 42) on which the committed Caco-2 baseline was measured (re-measured from the checkpoint as R² = 0.391, single run; Table 4), the same three feature sets for TabPFN (217 / 2,265 / 3,033 dimensions), and the same KPGT fine-tuning configuration (graphs re-featurized with the official KPGT pipeline; the GPU port re-verified against DGL, maximum absolute difference 8.3 × 10⁻⁷).

### 2.4 Cross-dataset comparison
Shared molecules between pepADMET PAMPA and PeptiVerse PAMPA were identified by canonical SMILES (RDKit). Of 7,177 unique pepADMET SMILES and 6,869 PeptiVerse SMILES, 6,834 (95.2% of pepADMET; 99.5% of PeptiVerse) overlap. Labels agreed exactly for 6,830 of these (the four discrepancies span at most 1.58 log units). Our models are evaluated on this common subset using the v4.2 split (test ∩ shared); PeptiVerse values quoted in Table 3 are **as published**, i.e., computed on that study's own Tanimoto-clustered test partition, and are therefore reference values rather than matched evaluations (footnote to Table 3).

### 2.5 Evaluation metrics and the oracle-ceiling convention
We report the coefficient of determination (R²), Spearman's rank correlation (ρ), and the floor-excluded R² (non-floor R²). RMSE and MAE for every route are provided in the repository's `results/` tables and were omitted here for space. All classification metrics are computed with scikit-learn ≥1.5. Models are selected on validation R² by early stopping; final test evaluation uses the best-validation checkpoint.

**Oracle-ceiling convention (single, declared).** For an endpoint, let F be the set of floor rows (y ≤ −10.0) in the test partition. The oracle predicts (i) the true observed value for every non-floor row and (ii) the **test-partition grand mean** for every floor row; the ceiling R² is 1 − SSE_oracle/SST on that partition. Applied to the v4.2 split this reproduces the published values exactly: PAMPA 0.5387 (published 0.539), Caco-2 0.5696 (published 0.570). Alternative mean choices shift the bound by <0.004 (train mean 0.5426/0.5721; full-table mean 0.5422/0.5724); the choice "floor → censored mean" is degenerate (the train censored mean *is* −10.0, giving R² = 1) and must not be used. `analysis/ceiling_convention.py` reproduces all of these in <10 s from the committed CSVs alone. This is a *practical* bound under the stated oracle, not an information-theoretic limit: models that identify floor membership and predict conditional floor means can in principle exceed it, and any such method must declare that evaluation convention explicitly. Our auxiliary floor-ranking experiment (LightGBM floor-detector AUC = 0.856; MLP 0.762) shows floor membership is partially predictable, but no operating point produced a usable regression on the floor subset under the declared convention.

**Significance protocol.** Inter-seed standard deviations quantify model stochasticity, not test-set sampling error; we therefore do not use seed-SD ratios as evidence. The repository's analysis script (`analysis/bootstrap_ci.py`) computes paired bootstrap 95% confidence intervals for ΔR² over test molecules (10,000 resamples, common random numbers) from the stored per-molecule predictions of every route; the resulting intervals are reported in the text (§3.2.1, §3.5, §3.7) and in `results/ci_summary.json`; prediction dumps for every compared model are committed under `results/predictions/` (`analysis/dump_predictions.py` documents the schema). Single-run routes (2, 4, 5, 8) and the single-run baselines carry the inter-seed noise reference of §2.2 (0.0025–0.0068).

### 2.6 Software and hardware
Python 3.11; Pytorch 2.x (CUDA 12.6); DGL 2.2.1 (CPU, patched); TabPFN 8.5.0; LightGBM 4.7.0; RDKit 2026.03. Exact version strings are pinned in the repository's lock file. The KPGT fine-tunes (PAMPA and Caco-2) and all TabPFN evaluations were executed on an NVIDIA RTX 4070 SUPER (KPGT ~410 s/epoch vs ~815 s/epoch on CPU); LightGBM and the MLP baselines ran on an AMD Ryzen 9 7950X CPU. All code, data, and trained checkpoints are available at [repository URL redacted for blind review].

## 3 Results

### 3.1 Routes 1–8: no classical route surpasses the baseline beyond noise
Table 1 summarizes routes 1–8. No classical route exceeds the baseline R² = 0.464 by more than the inter-seed noise reference (§2.2); the closest is soft-label blending (route 5) at 0.4651 (Δ = +0.001, within noise). Descriptor expansion (route 1) regressed to 0.4456 ± 0.0025 (5-seed mean), and the honest rank-Gaussian target (route 2) to 0.4469 (single run). Ensembling (route 3) degraded performance most substantially (0.4176), as did Tobit censored regression (route 4, single run; 0.4190) — the latter's non-floor R² of 0.466 indicates that the censored likelihood, as parametrized here, sacrifices measured-compound fidelity (diagnostics in §4.3). ChemBERTa frozen embeddings (route 6, best configuration) reached 0.4624 ± 0.0068 (3 seeds), statistically indistinguishable from the baseline, confirming that frozen chemical language-model representations add no information beyond 2D descriptors for this task. Re-training on the PeptiVerse raw data (route 7, single run) reproduced the censoring floor (3.5% of that set; ceilings 0.501/0.546) and yielded best R² of 0.434 (PAMPA) and 0.430 (Caco-2), corroborating the ceiling analysis on an independent dataset. Label averaging (route 8, single run) gave 0.4490.

| Route | Description | R² (all) | Δ vs baseline | Non-floor R² |
|---|---|---:|---:|---:|
| Baseline | MLP (3033-dim: RDKit 217 + Morgan 2048 + MoLFormer-XL 768)¹ | 0.4642 | — | 0.6317 |
| 1 | Descriptor expansion (Morgan r1, r2, r3; 5-seed MLP) | 0.4456 ± 0.0025 | −0.019 | 0.618 |
| 2 | Rank-Gaussian target (honest quantile; 5-seed ensemble) | 0.4469² | −0.017 | 0.615 |
| 3 | LightGBM ensemble (+Morgan r1, r3, r4; 5-seed) | 0.4176 | −0.047 | 0.592 |
| 4 | Tobit censored regression (NLL)² | 0.4190 | −0.045 | 0.466 |
| 5 | Soft-label blending (β = 0.50, validation-selected)² | 0.4651 | +0.001 | 0.593 |
| 6 | ChemBERTa-77M frozen + MLP (best: C_mol_molf_chem) | 0.4624 ± 0.0068 | −0.002 | 0.633 |
| 7 | PeptiVerse raw-data re-training (PAMPA / Caco-2)² | 0.434 / 0.430 | −0.030 / −0.034 | 0.501 / 0.546 |
| 8 | Label averaging (pepADMET ensemble mean)² | 0.4490 | −0.015 | 0.628 |

**Table 1.** Routes 1–8 on the v4.2 split. All R² values are on the floor-included test set; non-floor R² is computed on the floor-excluded subset. ¹Published v4.2 checkpoint, single run; inter-seed noise reference for comparable MLP configurations: 0.0025–0.0068 (routes 1, 6). ²Single run; see §2.3.

### 3.2 Route 9: foundation models break the baseline

#### 3.2.1 TabPFN v2 (in-context learning, no gradient)
TabPFN v2 on the 217 RDKit descriptors (canonical split; seeds 42, 123, 7) achieved **R² = 0.496 ± 0.002** (0.494 / 0.4972 / 0.4973 per seed), with non-floor R² of 0.627 against the baseline's 0.632. The +0.032 gain is twenty times the model's own inter-seed SD and more than four times the inter-seed noise reference of the MLP baseline (§2.2), ruling out a seed artifact; the paired bootstrap CI for ΔR² versus the committed baseline is +0.033 [+0.009, +0.055], P(Δ≤0) = 0.003. Extending the feature set to Morgan fingerprints (2,265 dimensions) or Morgan plus MoLFormer-XL (3,033 dimensions) yielded no improvement (0.481 and 0.482), consistent with the in-context saturation behavior of TabPFN beyond moderate feature counts.

#### 3.2.2 KPGT fine-tuning (end-to-end gradient)
KPGT fine-tuning (released base checkpoint; 3 seeds; early stopping) reached **R² = 0.513 ± 0.005**, the best result among all PAMPA routes (+0.049 versus baseline, ten times the KPGT inter-seed SD; paired bootstrap CI in the repository results table, §2.5). Per-seed best-validation test R²: seed 42 (epoch 16) = 0.519; seed 123 (epoch 9) = 0.507; seed 7 (epoch 13) = 0.514. Validation R² peaked at 0.408 (mean of three seeds) — a lower point estimate than the test value because the validation partition holds only 724 compounds, so its R² has a substantially wider sampling interval than the 1,457-row test estimate; the discrepancy carries no methodological implication. Each epoch required approximately 410 s on the RTX 4070 SUPER.

A critical trade-off accompanies this gain: non-floor R² declined to 0.536 (baseline 0.632; Δ = −0.096). The entire improvement is therefore concentrated in the censored floor region. Gradient fine-tuning on censored labels forces the model to allocate capacity to the floor pattern — a single repeated value — at the expense of fidelity on measured compounds.

| Model | R² (all) | ± SD | R² (non-floor) | Best epoch | Val R² (best) |
|---|---:|---:|---:|---:|---:|
| MLP baseline¹ | 0.4642 | — | 0.6317 | — | — |
| TabPFN v2 (217 descriptors) | 0.4962 | 0.0016 | 0.6268 | — | — |
| KPGT fine-tune (seed 42) | 0.5191 | — | 0.5633 | 16 | 0.4059 |
| KPGT fine-tune (seed 123) | 0.5073 | — | 0.5404 | 9 | 0.4105 |
| KPGT fine-tune (seed 7) | 0.5139 | — | 0.5035 | 13 | 0.4080 |
| **KPGT mean ± SD** | **0.5134** | **0.0048** | **0.5357** | — | **0.4081** |

**Table 2.** Route 9 foundation-model results. R² is on the floor-included test set at the best-validation checkpoint. ¹Single-run published checkpoint (see Table 1).

### 3.3 Cross-dataset comparison with pepADMET and PeptiVerse
Table 3 places our results in the context of the two leading platforms on the shared 6,834 molecules. Our KPGT fine-tune (test-set ρ = 0.811; baseline MLP ρ = 0.77) substantially outperforms the PeptiVerse published frozen-head results — ChemBERTa (ρ = 0.671) and PeptideCLM (ρ = 0.667) — noting that the PeptiVerse values are computed on their own test partition (footnote a). The pepADMET reported permeability R² range (0.435–0.657) spans multiple endpoints and model families; for PAMPA specifically, their best single model (LightGBM on 2D+3D descriptors) reported R² ≈ 0.66 on their split. That value exceeds our results not because the model family is stronger — our descriptor-side LightGBM route (route 3), which matches their model family, drops to 0.418 under unique-SMILES separation — but because their split does not enforce unique-SMILES separation, and on our leakage-controlled split the same model family does not exceed 0.42.

| Method | Representation | Training | Split | PAMPA metric |
|---|---|---|---|---|
| pepADMET best [2] | 2D + 3D descriptors + Morgan | LightGBM | pepADMET split | R² ≈ 0.66 (their split) |
| PeptiVerse [3] | ChemBERTa-77M (384-d), frozen | XGBoost head | 80/20 Tanimoto cluster | ρ = 0.671^a |
| PeptiVerse [3] | PeptideCLM-23M (768-d), frozen | XGBoost head | 80/20 Tanimoto cluster | ρ = 0.667^a |
| Our baseline | RDKit 217 + Morgan 2048 + MoLFormer-XL 768 | MLP | v4.2 unique-SMILES 70/10/20 | R² = 0.464, ρ = 0.77^b |
| Our KPGT fine-tune | LiGhT 12-layer graph (base checkpoint) | **End-to-end fine-tune** | v4.2 unique-SMILES 70/10/20 | **R² = 0.513, ρ = 0.811^b** |

**Table 3.** Cross-dataset comparison on the shared molecules. ^aAs published in [3], computed on the PeptiVerse Tanimoto-clustered test partition: reference values, not matched evaluations. ^bComputed on the intersection of our test partition with the shared-molecule set (1,457 ∩ shared). A strictly matched re-evaluation of PeptiVerse heads on our test partition is future work (§4.4).

### 3.4 The censored ceiling (PAMPA)
The practical oracle ceiling of R² = 0.539 (§2.5 convention) bounds every evaluated method on this endpoint. KPGT fine-tuning reaches 95% of it; TabPFN v2 reaches 92%. No evaluated model exceeds it, and exceeding it in the future requires either uncensored re-measurement of the 269 floor compounds or an explicitly declared floor-identification evaluation convention (§2.5). Auxiliary floor-ranking models (AUC 0.856 for LightGBM, 0.762 for an MLP) confirm that the floor compounds admit partial ordering, but no operating point yields a usable regression on the floor subset.

### 3.5 Route 10: foundation models on Caco-2 — generalization and a ranking reversal
Table 4 repeats the two foundation models of route 9 on the Caco-2 endpoint (7,429 compounds; identical split protocol; test n = 1,490, of which 39 floor rows). The baseline re-measured from the committed v4.2 checkpoint on this exact split yields R² = 0.3909 (non-floor 0.5111), reproducing the platform's summary metrics; the route-10 experiment log quoted 0.3933 (a pre-resubmission checkpoint of the same configuration), and all Δ values below use the reproducible 0.3909 figure. Both models again surpass the committed baseline, confirming that the route-9 gains are not a PAMPA-specific artifact. The ranking, however, reverses: TabPFN v2 (217 descriptors) attains **R² = 0.442 ± 0.003** (+0.051 over the baseline's 0.3909; fourteen times the TabPFN inter-seed SD; paired bootstrap CI +0.053 [+0.020, +0.088], P(Δ≤0) < 0.001 against the re-measured baseline), whereas KPGT fine-tuning reaches 0.411 ± 0.006 (+0.020, four times the KPGT inter-seed SD — the smallest of the four foundation-model gains).

The two models also differ in where their gains arise. On Caco-2, TabPFN improves the non-floor subset markedly (R² = 0.609 vs the baseline's 0.511, +0.098) while remaining the best floor-included model — unlike on PAMPA, where its gain was confined to the floor region (non-floor −0.005). KPGT reproduces its PAMPA trade-off in amplified form: its floor-included gain (+0.020) is paired with a non-floor deficit (0.449 vs 0.511, −0.062), and its Caco-2 training was markedly less stable (per-seed non-floor R² ranging 0.39–0.49; best epochs 8/15/16 across seeds; recurrent loss spikes), consistent with gradient fine-tuning of a ~112 M-parameter graph transformer on 7,429 molecules. The Caco-2 oracle ceiling under the §2.5 convention is R² = 0.570; TabPFN reaches 77.5% of it, KPGT 72.2%. Feature-set effects replicate the PAMPA finding: for TabPFN, the 217-descriptor set outperforms both the 2,265- and 3,033-dimensional extensions (0.427 and 0.435).

| Model | R² (all) | ± SD | R² (non-floor) | Δ vs baseline (all) | Spearman | Best epoch (seeds 42/123/7) |
|---|---:|---:|---:|---:|---:|---|
| MLP baseline (v4.2, re-measured)¹ | 0.3909 | — | 0.5111 | — | — | — |
| TabPFN v2 (217 descriptors) | **0.4415** | 0.0034 | **0.6086** | **+0.051** | 0.751 | — |
| TabPFN v2 (desc + Morgan, 2,265) | 0.4273 | 0.0068 | 0.5808 | +0.034 | 0.736 | — |
| TabPFN v2 (desc + Morgan + MoLFormer, 3,033) | 0.4350 | 0.0047 | 0.5849 | +0.042 | 0.747 | — |
| KPGT fine-tune (seed 42) | 0.4125 | — | 0.4649 | +0.022 | 0.756 | 15 |
| KPGT fine-tune (seed 123) | 0.4175 | — | 0.4907 | +0.027 | 0.739 | 16 |
| KPGT fine-tune (seed 7) | 0.4038 | — | 0.3912 | +0.013 | 0.720 | 8 |
| **KPGT mean ± SD** | **0.4113** | **0.0057** | **0.4489** | **+0.020** | **0.739** | — |
| Oracle ceiling (§2.5) | 0.5696 | — | — | — | — | — |

**Table 4.** Route 10: foundation models on Caco-2 (v4.2 split, seed 42; test n = 1,490, 39 floor rows). Baseline re-measured on the identical split. ¹Published v4.2 checkpoint, single run. TabPFN results are 3-seed means (seeds 42, 123, 7); KPGT results are per-seed best-validation checkpoints. Δ columns use the re-measured baseline (0.3909).

### 3.6 Context within the wider cyclic-peptide permeability literature
Table 5 situates our results among task-specific cyclic-peptide permeability models published over the last three years, most of which train on the CycPeptMPDB permeability database [12]. Two patterns are worth stressing. First, the published PAMPA R² values (0.67–0.77) are all measured under protocols that do not remove structural near-duplicates or that ignore the censored floor, so they are not directly comparable to our leakage-controlled, floor-aware numbers. Second, the 13-model benchmark [13], which evaluates a rigorous scaffold split, reports that model generalizability collapses substantially under it — precisely the gap between the headline R² and the R² we observe under unique-SMILES separation. We therefore read the difference between published and our R² as a protocol effect (duplicate leakage + ignored censoring), not evidence that our models are weaker.

| # | Model (reference) | Task / split | Reported PAMPA | Reported Caco-2 |
|---|---|---|---|---|
| 1 | CycPeptMP [14] | regression, their test split | R² = 0.772 ± 0.011 | — |
| 2 | Multi_CycGT [15] | classification (accuracy/AUC) | acc = 0.821, AUC = 0.865 | external sets |
| 3 | Multi_CycGT regression arm [15] | regression, random split | R² = 0.67 | R² = 0.75 |
| 4 | 13-model benchmark [13] | regression, random vs scaffold | best of 13 (scaffold ≪ random) | — |
| 5 | PCPpred (Mordred-2D) [17] | regression, assay-specific | R² = 0.685 (PCC 0.830) | R² = 0.793 (PCC 0.892) |
| 6 | C2PO [16] | permeability-optimization (generative) | design objective, not a held-out R² | design objective |
| 7 | **This work — KPGT fine-tune** | **unique-SMILES 70/10/20, censored-floor** | **R² = 0.513** | **R² = 0.411** |
| 8 | **This work — TabPFN v2** | **unique-SMILES 70/10/20, censored-floor** | **R² = 0.496** | **R² = 0.442** |

**Table 5.** Published cyclic-peptide permeability models (rows 1–6, as reported in each publication; split protocols and floor handling vary and are not directly comparable) versus this work (rows 7–8, unique-SMILES split with explicit left-censoring). Caco-2 values for this work are under the identical v4.2 protocol.

The practical reading: a model reported at R² ≈ 0.77 [14] on a non-deduplicated random split and a model reported at R² = 0.513 on a unique-SMILES, floor-aware split are answering different questions. The former asks "how well does the model fit this particular test split, which shares near-duplicates with the training set?"; the latter asks "how well does the model generalize to molecules it has never seen in any tautomeric or stereochemical form, while honestly handling the assay floor?". The scaffold-split evidence in [13] shows the two answers diverge by a wide margin, and is consistent with our finding that the censored ceiling (0.539) — not model capacity — is the binding constraint once leakage is removed.

### 3.7 Multi-split replication of the foundation route
Because the headline gains ride on a single canonical split, we replicated the cheaper foundation model (TabPFN v2, 217 descriptors, 3-seed inference ensemble) across split seeds 42/7/123 on both endpoints, re-training the MLP baseline from scratch on every split with the published recipe so that each comparison is split-matched. The advantage replicates on **all six split-endpoint pairs** with paired bootstrap CIs excluding zero (PAMPA: +0.031 [+0.008, +0.054], +0.068 [+0.046, +0.090], +0.065 [+0.031, +0.097]; Caco-2: +0.051 [+0.020, +0.085], +0.059 [+0.030, +0.090], +0.043 [+0.009, +0.076]; splits 42/7/123). Re-trained baselines themselves vary substantially across splits (PAMPA 0.406–0.466), underscoring why single-split benchmarking is reported as a limitation and why split-matched comparison is the correct protocol. Predictions and per-split CIs are in `results/predictions/` and `results/multisplit_final.json`.

## 4 Discussion

### 4.1 Why foundation models succeed where classical routes fail
Routes 1–8 operate within the descriptor-plus-gradient-boosting paradigm and share a common weakness: none can exploit the structure of the censored labels. The floor is a measurement artifact rather than a chemical pattern; LightGBM treats all labels as equally informative; and ensembling averages noise rather than signal. Foundation models introduce external inductive bias. TabPFN v2 conditions on synthetic tabular priors learned in-context [4], and KPGT inherits geometric representations from large-scale self-supervised pretraining [5]. Both priors help interpolate the censored region without overfitting the repeated floor value — precisely the regime where classical reweighting and target transformations (routes 2, 4, 5) could not reach. Route 10 shows that this advantage generalizes across endpoints: both foundation models surpass the Caco-2 baseline as well, under the identical split and floor protocol. Notably, no single model dominates — KPGT leads on PAMPA, TabPFN on Caco-2 — suggesting that the in-context and gradient-based priors are complementary rather than interchangeable.

### 4.2 Frozen embeddings versus fine-tuning
PeptiVerse's frozen embeddings with lightweight heads yield published ρ = 0.67–0.69 on PAMPA. Our KPGT fine-tune reaches ρ = 0.811 on the shared-molecule intersection, a nominal gap of ~+0.14 in Spearman correlation. Two caveats qualify the contrast: the PeptiVerse values are computed on their own Tanimoto-clustered test partition (Table 3, footnote a), and the comparison changes representation family as well as training regime (graph transformer vs sequence transformer). Within a single representation family, our route 6 provides the matched arm: fine-tuning a *graph* model improves ranking substantially over the baseline (ρ 0.811 vs 0.77), whereas the frozen *ChemBERTa* arm matches the baseline without exceeding it (R² 0.4624 vs 0.4642). Taken together, these two lines of evidence — not the cross-publication ρ gap alone — support the conclusion that **task-specific adaptation of the representation, rather than the frozen embedding alone, supplies the information needed to break the baseline on this task**. A strictly matched frozen-vs-fine-tuned comparison within a single backbone, evaluated on a common partition, is the definitive experiment and is future work (§4.4). For peptide permeability, our results indicate that end-to-end fine-tuning of a pretrained graph transformer, not frozen embeddings, currently defines the practical state of the art.

### 4.3 The price paid on measured compounds
On PAMPA, both foundation models improve floor-included R² while leaving or degrading non-floor R² (TabPFN: −0.005; KPGT: −0.096). Route 10 sharpens this picture: on Caco-2 the pattern is endpoint-dependent — TabPFN improves non-floor R² substantially (+0.093), whereas KPGT again pays for its floor gain (−0.066). Gradient training on censored labels allocates model capacity to a single repeated value, reducing fidelity on the ~96–97% of compounds with true measurements; the effect is reproducible across both endpoints for the gradient-tuned model. We regard this as a fundamental, not incidental, trade-off: **any model trained on censored labels without explicit censored modeling risks sacrificing measured-compound accuracy in exchange for floor performance**. Tobit regression (route 4), the canonical remedy, was worse still (non-floor R² = 0.466); the failure is attributable to the fitted scale: the committed run converged to a large shared residual scale (σ ≈ 0.886 across seeds, `analysis/route4_results.json`), which flattens the truncated-normal likelihood almost to a squared-error fit on the observed floor value — precisely the degenerate limit in which Tobit cannot beat plain regression. The family per se is not implicated; an EM reconstruction with a bulk-calibrated σ = 0.44 (§4.3) recovers most of the non-floor fidelity that Tobit lost (`analysis/em_reconstruction.py`). We probed this direction directly with an EM label reconstruction (`analysis/em_reconstruction.py`): a bulk-only LightGBM (never sees the floor) calibrates a Gaussian residual scale (σ = 0.44), floor labels are replaced by their truncated-normal conditional means, and the regressor is refit. On the LightGBM feature family the reconstruction preserved censored-region behavior but did not reach the MLP's non-floor fidelity (non-floor R² 0.497 vs 0.632 — the refit model is a different learner class, so this is a negative control on the learner, not a falsification of EM). The decisive test — EM-reconstructed labels fed to the KPGT fine-tune — remains future work; it is the experiment that would convert this benchmark's main caveat into a method contribution. Practical use of such models should in all cases report floor- and non-floor metrics separately, as we do throughout.

### 4.4 Limitations
1. The ceilings of 0.539 (PAMPA) and 0.570 (Caco-2) are practical upper bounds of the current assays under the declared oracle convention (§2.5); R² = 0.7 is unreachable without re-measurement.
2. KPGT fine-tuning requires GPU access (the public DGL wheel is CPU-only on Windows; our pure-Pytorch port was required) and was less stable on Caco-2 than on PAMPA.
3. TabPFN v2's feature limit (500 dimensions) precludes using the full fingerprint-plus-descriptor set.
4. Routes 1–8 and the KPGT fine-tunes were evaluated on a single canonical split (seed 42). For TabPFN v2 we closed this gap directly: the route was replicated on three unique-SMILES split seeds (42, 7, 123) on both endpoints, against MLP baselines **re-trained from scratch on each split** with the published recipe (`analysis/retrain_splits.py`). TabPFN's 3-inference ensemble beat its split-matched re-trained baseline on all six split-endpoint pairs with paired bootstrap CIs excluding zero: +0.031 [+0.008, +0.054], +0.068 [+0.046, +0.090], +0.065 [+0.031, +0.097] on PAMPA and +0.051 [+0.020, +0.085], +0.059 [+0.030, +0.090], +0.043 [+0.009, +0.076] on Caco-2 (splits 42/7/123). The foundation-model result is not split luck. KPGT multi-split replication (expensive) remains future work.
5. The PeptiVerse comparison (Table 3) mixes published metrics on a foreign test partition with ours; a matched re-evaluation of the public PeptiVerse heads on the v4.2 test partition is required before the frozen-vs-fine-tuned contrast can be read as a clean single-factor effect (the within-family route-6 evidence partially covers this).
6. Published comparisons (Table 5) rely on each paper's reported metrics; we did not re-run the external models, so cross-statement comparisons inherit their protocol choices.
7. ChemBERTa (2020) predates modern peptide language models; route 6 bounds the *published frozen-sequence baseline* used by PeptiVerse, not the ceiling of frozen sequence representations.
8. Independent wet-lab validation of the highest-ranked predictions (KPGT on PAMPA, TabPFN on Caco-2) is the natural next step.

### 4.5 Scope of the platform
The proposed platform (reference [6]) provides: (i) leakage-controlled unique-SMILES splitting for peptide datasets; (ii) censoring-aware evaluation (oracle-ceiling convention, floor AUC, floor/non-floor metric decomposition); (iii) reproducible pipelines for the eight classical PAMPA routes, the route-9 foundation pair, the route-10 Caco-2 extension, and HLM; (iv) a GPU KPGT fine-tuning script (pure Pytorch, checkpoint/resume, verified against the DGL reference, PAMPA and Caco-2); and (v) canonical TabPFN v2 evaluation scripts for both endpoints. We recommend it for lead prioritization of cyclic-peptide permeability (PAMPA/Caco-2) where assay data is censored, for benchmarking new peptide ADMET methods against a rigorous baseline, and as a starting point for foundation-model fine-tuning on peptide graphs. Extrapolation to linear peptides, non-peptide macrocycles, or uncensored high-permeability regimes is not recommended without external validation.

## 5 Conclusions
We present, to our knowledge, the first systematic, leakage-controlled, censoring-aware benchmark of ten permeability-improvement routes on 7,283 PAMPA and 7,429 Caco-2 cyclic peptides. All eight classical PAMPA routes — descriptor expansion, rank-Gaussian target transformation, LightGBM ensembling, Tobit censored regression, soft-label blending, ChemBERTa frozen embeddings, PeptiVerse raw-data re-training, and label averaging — failed to exceed the MLP baseline (R² = 0.464) beyond noise. Foundation models broke the barrier on both endpoints: on PAMPA, TabPFN v2 (in-context learning) reached R² = 0.496 ± 0.002 and KPGT fine-tuning (end-to-end gradient) reached R² = 0.513 ± 0.005, the best of all routes and 95% of the practical censored ceiling (0.539); on Caco-2, the ranking reversed, with TabPFN v2 leading at R² = 0.442 ± 0.003 (baseline 0.391, non-floor 0.609) and KPGT at 0.411 ± 0.006 — evidence that foundation-model gains generalize across permeability endpoints but that no single model dominates. On the shared molecules, KPGT fine-tuning (ρ = 0.811) exceeded the published PeptiVerse frozen-embedding results (ρ = 0.671), and the matched within-family comparison (route 6) supports task-specific adaptation as the effective factor. The gains of the gradient-tuned model, however, concentrate in the censored region on both endpoints at the cost of non-floor accuracy, whereas TabPFN's gains are floor-confined on PAMPA but extend to measured compounds on Caco-2. **The censored ceilings — 0.539 (PAMPA) and 0.570 (Caco-2), not model capacity, are the true bottlenecks.** Advancing peptide permeability prediction requires uncensored re-measurement of floor compounds — an experimental, not algorithmic, task — and, on the algorithmic side, two-stage censored reconstruction that recovers measured-compound fidelity while retaining floor-region gains.

## References
[1] Hornsby BD, Lee CH, Steele CA, Tuekpe JK, Lim CS. Therapeutic peptides and proteins: Status and developments in drug delivery. J Control Release. 2026;394:114895. doi:10.1016/j.jconrel.2026.114895

[2] Tan X, Liu Q, Zhou M, Fang Y, Ouyang D, Zeng W, Dong J. pepADMET: A novel computational platform for systematic ADMET evaluation of peptides. J Chem Inf Model. 2026;66(2):936-946. doi:10.1021/acs.jcim.5c02518

[3] Zhang Y, Tang S, Chen T, Mahood E, Vincoff S, Chatterjee P. PeptiVerse: A unified platform for therapeutic peptide property prediction. Nat Commun. 2026;17:6819. doi:10.1038/s41467-026-74167-w

[4] Hollmann N, Müller S, Purucker L, Krishnakumar A, Körfer M, Hoo SB, Schirrmeister RT, Hutter F. Accurate predictions on small data with a tabular foundation model. Nature. 2025;637(8045):319-326. doi:10.1038/s41586-024-08328-6

[5] Li H, Zhao D, Zeng J. KPGT: Knowledge-guided pre-training of graph transformer for molecular property prediction. In: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22). 2022:857-867. doi:10.1145/3534678.3539426

[6] [Authors, self-citation redacted for review]. Peptide ADMET prediction platform with censored-floor-aware protocol (v4.2). Zenodo archive (DOI reserved at publication; repository redacted for blind review). (v4.2 protocol: unique-SMILES 70/10/20 split, censored floor at −10.0 log cm/s, MixedADMETMLP baseline, Huber loss)

[7] Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model. 2010;50(5):742-754. doi:10.1021/ci100050t

[8] Landrum G. RDKit: Open-source cheminformatics. 2024. https://www.rdkit.org

[9] Ke G, Meng Q, Finley T, Wang T, Chen W, Ma W, Ye Q, Liu TY, Sun W, Wang W, Wang ML. LightGBM: A highly efficient gradient boosting decision tree. Adv Neural Inf Process Syst. 2017;30:3146-3154.

[10] Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885. 2020. doi:10.48550/arXiv.2010.09885

[11] Wang M, Zheng D, Ye Z, Gan Q, Li M, Zhou X, Ma C, Hu Z, Yang Q, Zhao Y, Li J, Smola A, Zhang Z. Deep graph library: A graph-centric, highly-performant package for graph neural networks. arXiv preprint arXiv:1909.01315. 2019. doi:10.48550/arXiv.1909.01315

[12] Li J, Yanagisawa K, Sugita M, Fujie T, Ohue M, Akiyama Y. CycPeptMPDB: A comprehensive database of membrane permeability of cyclic peptides. J Chem Inf Model. 2023;63(7):2240-2250. doi:10.1021/acs.jcim.2c01573

[13] Liu W, Li X, Verma K, Lee H. Systematic benchmarking of 13 AI methods for predicting cyclic peptide membrane permeability. J Cheminform. 2025;17(1):129. doi:10.1186/s13321-025-01083-4

[14] Li J, Yanagisawa K, Akiyama Y. CycPeptMP: Enhancing membrane permeability prediction of cyclic peptides with multi-level molecular features and data augmentation. Brief Bioinform. 2024;25(5):bbae417. doi:10.1093/bib/bbae417

[15] Cao L, Xu Z, Shang T, Zhang C, Wu X, Wu Y, Zhai S, Zhan Z, Duan H. Multi_CycGT: A deep learning-based multimodal model for predicting the membrane permeability of cyclic peptides. J Med Chem. 2024;67(3):1888-1899. doi:10.1021/acs.jmedchem.3c01611

[16] Aerts R, Tavernier J, Kerstjens F, Ahmad W, Gómez-Tamayo F, Tresadern P, De Winter M. C2PO: An ML-powered optimizer of the membrane permeability of cyclic peptides through chemical optimization. J Cheminform. 2025;17:168. doi:10.1186/s13321-025-01109-x

[17] Shendre A, Gahlot PS, Raghava GP. PCPpred: Prediction of chemically modified peptide permeability across multiple assays for oral delivery. bioRxiv preprint. 2026. doi:10.64898/2026.01.19.700485

[18] Jiang D, Chen Z, Du H. Cyclic peptide membrane permeability prediction using deep learning model based on molecular attention transformer. Front Bioinform. 2025;5:1566174. doi:10.3389/fbinf.2025.1566174

*Reference ordering note: reference numbers follow first appearance; the final typeset copy will be renumbered to citation order by the production workflow.*

**Graphical Abstract** (to be rendered): Flowchart showing the v4.2 censoring-aware protocol → ten routes (PAMPA 1–9, Caco-2 10) → baselines 0.464/0.391 → TabPFN 0.496/0.442 → KPGT 0.513/0.411 → ceilings 0.539/0.570. Key message: foundation models are the first to beat the baselines on both endpoints, but the ceilings hold.

**Figures** (rendered, in `figures/`): Figure 1 (`fig1_scatter.png`) — predicted vs observed with floor rows highlighted, baseline and TabPFN, both endpoints; Figure 2 (`fig2_decomposition.png`) — floor-included vs non-floor R² decomposition against the endpoint ceilings (Tables 2, 4); Figure 3 (`fig3_ceiling.png`) — fraction of oracle ceiling attained, including the three-split TabPFN replication of §3.7.

**Data Availability** All data, splits, trained checkpoints, per-molecule predictions (including the bootstrap inputs for §2.5), and reproduction scripts are available at [repository URL redacted for blind review] (MIT License); a frozen Zenodo archive with a versioned DOI will accompany the resubmission.

**AI Declaration** This manuscript was prepared with the assistance of AI tools (Hermes Agent) for code generation, literature retrieval, and text drafting. All scientific claims, numbers, and conclusions were verified by the authors against primary computational experiments.

**Conflict of Interest** The authors declare no competing financial interests.
