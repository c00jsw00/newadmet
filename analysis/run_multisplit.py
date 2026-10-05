import numpy as np, pandas as pd, time, sys
sys.path.insert(0,'.')
import feature_extractor
from sklearn.metrics import r2_score
FLOOR=-10.0
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
out={}
for csvp,col,ep in [('base.csv','PAMPA_MDCK','PAMPA'),('caco2.csv','Caco2','CACO2')]:
    df=pd.read_csv(csvp); sm=df['smiles'].astype(str).tolist(); y=df[col].to_numpy(np.float64)
    uniq,inv=np.unique(np.asarray(sm,dtype=object),return_inverse=True)
    desc=feature_extractor.molecule_features(sm)[:, :217]
    for ssplit in (42,7,123):
        rng=np.random.default_rng(ssplit); perm=rng.permutation(len(uniq))
        n_tr=int(round(len(uniq)*.7)); n_va=int(round(len(uniq)*.1))
        tr_u=set(perm[:n_tr].tolist()); te_u=set(perm[n_tr+n_va:].tolist())
        tr=np.array([i for i in range(len(y)) if inv[i] in tr_u])
        te=np.array([i for i in range(len(y)) if inv[i] in te_u])
        ps=[]
        for s in (42,123,7):
            m=TabPFNRegressor(model_path=TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path,
                              n_estimators=4, device='cuda', random_state=s, ignore_pretraining_limits=True)
            m.fit(desc[tr],y[tr]); ps.append(np.asarray(m.predict(desc[te]),dtype=np.float64))
        pen=np.mean(ps,axis=0); fl=y[te]<=FLOOR+1e-9
        r2a=r2_score(y[te],pen); r2nf=r2_score(y[te][~fl],pen[~fl])
        sd=float(np.std([r2_score(y[te],p) for p in ps]))
        out[f'{ep}/split{ssplit}']=dict(r2_all=float(r2a),r2_all_seedsd=sd,r2_nonfloor=float(r2nf),
                                        n_test=int(len(te)),n_floor=int(fl.sum()))
        print(ep,'split',ssplit,': r2_all=%.4f (seed-SD %.4f) r2_nf=%.4f test=%d floor=%d'%(r2a,sd,r2nf,len(te),int(fl.sum())),flush=True)
import json; json.dump(out,open('multisplit_tabpfn.json','w'),indent=1)
