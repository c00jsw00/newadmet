# Literature reference values used to sanity-check the GLP-1 predictions

All values below were retrieved live during the run described in the parent
README; links were opened and the quoted sentences extracted from the page text.

| Quantity | Measured value (source) | Model prediction | Verdict |
|---|---|---|---|
| Native GLP-1(7-37) plasma t1/2 (human) | ~1.5–5 min (rapid DPP-4 clearance; "approximately 0.08 hours (4.8 minutes)" — PMC12632547; "intravenous half-life from 1 to 4-5 minutes" — ScienceDirect GLP-1 analog review) | 18.2 min (log10 t1/2 = 3.04) | right order of magnitude; ~4x overestimate — model in-distribution (5 sequence features beyond 5 sigma) |
| GLP-1 analog (exenatide) Caco-2 Papp, 3-day system | 3.1–7.8 x 10^-6 cm/s = logPapp -5.1 to -5.5 (Gupta et al. 2013, PMC3586668, Table 1) | logPapp -11.2 (≈ 7 x 10^-12 cm/s) | qualitative agreement ("not permeable at oral-relevant levels") but 5+ log units too low — MOLECULAR endpoints are out-of-domain for a 3.4-kDa linear peptide (train MW <= 1700) |
| GLP-1 analog Caco-2 Papp (Youn et al., cited in PMC3586668) | 0.5 +/- 0.2 x 10^-7 cm/s = logPapp ~ -7.3 | same as above | same conclusion |
| GLP-1 hemolysis | no published assay; circulating hormone active at pM-nM, non-hemolytic by physiology | p = 0.029 (non-hemolytic) | consistent, not verifiable quantitatively |

Sources (checked 2026-10-05):
- Gupta V, Doshi N. "Permeation of Insulin, Calcitonin and Exenatide across
  Caco-2 Monolayers: Measurement Using a Rapid, 3-Day System." PMC3586668.
- PMC12632547 (azapeptide GLP-1 analogue paper; quotes native GLP-1(7-37)
  plasma clearance ~0.08 h).
- UniProt P01275 (pro-glucagon): GLP-1(7-37) = residues 98-128,
  HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG — the exact sequence predicted here.
