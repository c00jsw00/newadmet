# newadmet

Revised manuscript v2.0 for the peptide ADMET benchmark study
(*Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation*),
revised in response to expert peer review of the previous draft
(`peptide-admet-jpa`).

## Contents
| File | Description |
|---|---|
| `peptide_admet_manuscript_jcim_v2.md` | Revised main manuscript (v2.0) |
| `peptide_admet_manuscript_jcim_blind_v2.md` | Anonymized copy for double-blind review (leak-checked) |
| `REVISION_NOTES.md` | Point-by-point reviewer-response / change log |

## Headline results (unchanged numerics; framing revised)
- PAMPA baseline R² = 0.464 → KPGT fine-tune **0.513 ± 0.005**, TabPFN v2 **0.496 ± 0.002**
- Caco-2 baseline R² = 0.393 → TabPFN v2 **0.442 ± 0.003**, KPGT 0.411 ± 0.006
- Oracle ceilings (declared convention, §2.5): 0.539 (PAMPA) / 0.570 (Caco-2)

## Key revisions vs v1.0
Corrected references (KPGT = KDD '22; TabPFN/ChemBERTa author lists; Table-5 citation map),
unified oracle-ceiling convention, bootstrap-CI significance protocol, re-scoped
frozen-vs-fine-tuned claims, blind-copy anonymity fix, ACS US-orthography pass, abstract ≤ 250 words.

MIT License (carried over from `peptide-admet-jpa`).
