# newadmet — Peptide ADMET Manuscript, Revised (v2.0)

Revised manuscript responding to expert peer review of the peptide ADMET benchmark
("Peptide ADMET Prediction: A Systematic Benchmark and Foundation Model Evaluation").

## Files
- `peptide_admet_manuscript_jcim_v2.md` — revised main manuscript (v2.0)
- `peptide_admet_manuscript_jcim_blind_v2.md` — anonymized copy for double-blind review
- `REVISION_NOTES.md` — point-by-point reviewer response mapping every change

## Summary of revisions (v1.0 → v2.0)
1. **References corrected (verified against CrossRef/arXiv):**
   - KPGT re-cited to its published venue: KDD '22, pp. 857–867, doi:10.1145/3534678.3539426
     (was miscited as an "ICML 2024 arXiv preprint", arXiv:2206.03364).
   - TabPFN reference corrected (Müller, Körfer; pages 319–326).
   - ChemBERTa author list corrected (Chithrananda S, Grand G, Ramsundar B).
   - Table 5 citation mapping repaired (CycPeptMP → [14], Multi_CycGT → [15],
     13-model benchmark → [13], PCPpred → [17], C2PO → [16]).
2. **Oracle-ceiling convention unified (§2.5):** one declared convention (oracle = true
   value on unmeasured rows + training-set censored mean on floor rows), the same equation
   applied to PAMPA (0.539) and Caco-2 (0.570); absolute "no model can exceed" wording
   replaced with "practical bound under the declared convention".
3. **Statistics framework (§2.5):** seed-SD ratios demoted to a noise reference; paired
   bootstrap CI protocol for ΔR² specified (script: `analysis/bootstrap_ci.py`); the
   incorrect "16×" and "13–20×" figures corrected (TabPFN 20×, KPGT-PAMPA 10×, KPGT-Caco-2 3×).
4. **Frozen-vs-fine-tuned claim re-scoped (§4.2):** cross-publication ρ gap labelled a
   reference comparison (footnote to Table 3); the within-family route-6 arm is presented
   as the matched evidence; matched frozen-vs-fine-tuned experiment named as future work.
5. **Leakage argument anchored to route 3 (§3.3):** the "same model family" claim now cites
   the LightGBM route (0.418) as the correct comparator for the pepADMET-published LightGBM.
6. **Route renaming/labelling:** route 7 is now "PeptiVerse raw-data re-training" (not
   cross-validation), with the 606-row Caco-2 subset limitation stated explicitly.
7. **Single-run routes flagged** in §2.3 and Table 1 footnotes; Tobit failure attributed to
   parametrization, not the Tobit family (§4.3), with diagnostics deferred to the repo.
8. **Validation-vs-test R² gap explained** (n=724 validation partition) in §3.2.2.
9. **Blind-copy leak fixed:** §4.5 now says "the proposed platform (reference [6])";
   automated check confirms no author/repo identifier survives in the blind copy.
10. **Editorial:** US orthography throughout (ACS style); "Pytorch" → "Pytorch";
    ADMET's E = elimination; abstract compressed to 246 words; "nine routes — eight
    classical and two foundation" contradiction removed; keywords trimmed to 6;
    repository URL updated throughout; replicate-row handling (7,283 rows / 7,177 unique
    SMILES) documented in §2.1; planned figures listed for resubmission.

## Honest caveats
- Bootstrap CIs: the protocol and the generating script location are documented, but the
  per-comparison interval values are to be filled from `results/` before resubmission —
  no interval numbers are asserted in the text that were not actually computed.
- The KPGT pretraining-corpus figure ("1.6 M molecules") was removed pending verification
  against the published KDD paper.
- Figure assets and the graphical abstract remain marked "to be rendered".

Source: revised from `peptide-admet-jpa` (main @ v1.0).
