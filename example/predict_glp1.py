#!/usr/bin/env python3
"""Worked example: four-endpoint ADMET prediction for GLP-1(7-37).

Peptide: human glucagon-like peptide-1, fragment 7-37 (P01275 residues 98-128)
    HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG          (31 residues)

Endpoints and how each row is scored:
  * PAMPA_MDCK, Caco2 (molecular modality): linear sequence is rendered to
    canonical SMILES with RDKit `MolFromFASTA`, featurized as
    2265-dim RDKit vector + 768-dim frozen MoLFormer-XL CLS embedding
    (precomputed with molformer_embed.py -> data/molformer/glp1_emb.npz),
    then scored by the committed v4.2 single-task MLP (models_v4/).
  * Hemolysis (binary probability) and Half_life (plasma t1/2, seconds):
    428-dim classical sequence vector + 1152-dim frozen ESMC-600M CLS
    embedding (data/esmc/glp1_esmc.pt), scored by the v4.1/v4.2 checkpoints.

Run:  python predict_glp1.py
Outputs: printed report + results/glp1_predictions.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admet_model import MixedADMETMLP
from feature_extractor import molecule_features, sequence_features

HERE = Path(__file__).resolve().parent
SEQ = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"
FLOOR = -10.0


def load_model(slug, endpoint):
    blob = torch.load(HERE / "models_v4" / slug / "admet_mlp.pt",
                      map_location="cpu", weights_only=False)
    m = MixedADMETMLP(input_dim=blob["input_dim"], endpoints=blob["endpoints"],
                      hidden=blob.get("hidden", (256, 128)),
                      dropout=blob.get("dropout", 0.25))
    m.load_state_dict(blob["state_dict"])
    m.eval()
    torch.serialization.add_safe_globals([StandardScaler])
    scaler = torch.load(HERE / "models_v4" / slug / "scaler.pt",
                        map_location="cpu", weights_only=False)
    def score(X):
        # manual scaling in float64 (raw RDKit descriptors can exceed float32),
        # followed by a +-30-sigma winsorizing cap.  The cap matters: the
        # information content of a content descriptor (Ipc) for a 31-mer is
        # ~49 sigma above the training ceiling (cyclic peptidomimetics of
        # 4-12 residues, max Ipc 4.1e36), so an uncapped linear head explodes.
        # Any feature beyond +-5 sigma is flagged out-of-distribution.
        X64 = np.asarray(X, dtype=np.float64)
        Z = (X64 - scaler.mean_) / scaler.scale_
        ood = int(np.count_nonzero(np.abs(Z) > 5.0))
        score.ood_features = ood
        Xs = np.clip(Z, -30, 30).astype(np.float32)
        with torch.no_grad():
            o = m(torch.from_numpy(Xs))[endpoint]
        o = torch.sigmoid(o).squeeze(1) if endpoint == "Hemolysis" else o.squeeze(1)
        return o.numpy()
    return score, blob["input_dim"]


def main():
    mol = Chem.MolFromFASTA(SEQ)
    assert mol is not None, "FASTA parse failed"
    smi = Chem.MolToSmiles(mol, canonical=True)
    mw = Descriptors.MolWt(mol)
    print(f"peptide      : GLP-1(7-37), {len(SEQ)} aa, MW {mw:.1f} Da")
    print(f"smiles       : {len(smi)} chars, canonical RDKit linear peptide\n")

    # ---- molecular endpoints ------------------------------------------------
    z = np.load(HERE / "data" / "molformer" / "glp1_emb.npz", allow_pickle=False)
    emb = np.asarray(z["emb"], dtype=np.float32)
    X_mol = np.concatenate([molecule_features([smi]).astype(np.float32), emb], axis=1)
    out = {}
    for slug, ep in (("pampa_mdck", "PAMPA_MDCK"), ("caco2", "Caco2")):
        score, dim = load_model(slug, ep)
        assert X_mol.shape[1] == dim, (X_mol.shape, dim)
        v = float(score(X_mol)[0])
        out[ep] = {"logPapp": v, "ood_features_gt5sigma": score.ood_features}
        ood = score.ood_features
        pct = "at/below assay floor" if v <= FLOOR + 0.5 else "above assay floor"
        print(f"{ep:11s}: logPapp = {v:8.3f}  ({pct}; {ood} features beyond +-5 sigma of the training distribution)")

    # ---- sequence endpoints --------------------------------------------------
    ze = np.load(HERE / "data" / "esmc" / "glp1_esmc.npz", allow_pickle=False)
    esm = np.asarray(ze["emb"], dtype=np.float32)
    X_seq = np.concatenate([sequence_features([SEQ]).astype(np.float32), esm], axis=1)
    for slug, ep in (("hemolysis", "Hemolysis"), ("half_life", "Half_life")):
        score, dim = load_model(slug, ep)
        assert X_seq.shape[1] == dim, (X_seq.shape, dim)
        v = float(score(X_seq)[0])
        out[ep+"_ood"] = score.ood_features
        if ep == "Hemolysis":
            out[ep] = v
            print(f"{ep:11s}: hemolytic probability = {v:6.3f}  "
                  f"({'predicted hemolytic' if v >= .5 else 'predicted non-hemolytic'} @0.5)")
        else:
            t = float(np.power(10.0, v))
            out[ep] = {"log10_t_half_s": v, "t_half_s": t}
            print(f"{ep:11s}: log10 t1/2 = {v:6.3f}  -> t1/2 = {t:,.0f} s ({t/60:.1f} min)")

    out["_meta"] = {"peptide": SEQ, "name": "GLP-1(7-37)", "mw": round(float(mw), 1),
                    "smiles_len": len(smi), "smiles": smi}
    rp = HERE / "results" / "glp1_predictions.json"
    rp.parent.mkdir(exist_ok=True)
    json.dump(out, open(rp, "w"), indent=1)
    print(f"\nwrote {rp}")


if __name__ == "__main__":
    main()
