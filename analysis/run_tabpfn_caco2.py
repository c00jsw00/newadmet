import numpy as np, pandas as pd, time, sys
sys.path.insert(0,'.')
import feature_extractor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

FLOOR=-10.0; SEED=42
df=pd.read_csv('caco2.csv'); sm=df['smiles'].astype(str).tolist(); y=df['Caco2'].to_numpy(np.float64)
uniq,inv=np.unique(np.asarray(sm,dtype=object),return_inverse=True)
desc = feature_extractor.molecule_features(sm)[:, :217]
rng=np.random.default_rng(SEED); perm=rng.permutation(len(uniq))
n_tr=int(round(len(uniq)*.7)); n_va=int(round(len(uniq)*.1))
tr_u=set(perm[:n_tr].tolist()); va_u=set(perm[n_tr:n_tr+n_va].tolist()); te_u=set(perm[n_tr+n_va:].tolist())
tr=np.array([i for i in range(len(y)) if inv[i] in tr_u]); te=np.array([i for i in range(len(y)) if inv[i] in te_u])
print('split rows train/test:',len(tr),len(te),flush=True)

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
t0=time.time(); preds={}
for seed in (42,123,7):
    m=TabPFNRegressor(model_path=TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path,
                      n_estimators=4, device='cuda', random_state=seed, ignore_pretraining_limits=True)
    m.fit(desc[tr], y[tr])
    preds[seed]=np.asarray(m.predict(desc[te]),dtype=np.float64)
    y_te=y[te]; nf=~(y_te<=FLOOR+1e-9)
    print(f'seed {seed}: r2_all={r2_score(y_te,preds[seed]):.4f} r2_nf={r2_score(y_te[nf],preds[seed][nf]):.4f} rho={spearmanr(y_te,preds[seed]).statistic:.4f}',flush=True)
np.savez('preds_tabpfn_v2_desc_caco2.npz', te=te, y=y, **{f'p{s}':preds[s] for s in preds})
print('elapsed',int(time.time()-t0),'s')
