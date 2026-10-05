import numpy as np, pandas as pd, torch, torch.nn as nn, sys, json
sys.path.insert(0,'.')
import feature_extractor
from admet_model import MixedADMETMLP
from sklearn.metrics import r2_score
torch.manual_seed(42); np.random.seed(42)
def split(smiles,seed):
    uniq,inv=np.unique(np.asarray(smiles,dtype=object),return_inverse=True)
    N=len(smiles); rng=np.random.default_rng(seed); perm=rng.permutation(len(uniq))
    n_tr=int(round(len(uniq)*.7)); n_va=int(round(len(uniq)*.1))
    ids={'tr':set(perm[:n_tr].tolist()),'va':set(perm[n_tr:n_tr+n_va].tolist()),'te':set(perm[n_tr+n_va:].tolist())}
    return tuple(np.array([i for i in range(N) if inv[i] in ids[k]],dtype=np.int64) for k in ('tr','va','te'))
def train_mlp(Xtr,ytr,Xva,yva,input_dim,head):
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler().fit(Xtr)
    mlp=MixedADMETMLP(input_dim=input_dim,endpoints=[head],hidden=(256,128),dropout=0.25)
    opt=torch.optim.Adam(mlp.parameters(),lr=1e-3,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=4)
    lossf=nn.HuberLoss(delta=1.0)
    Xt=torch.from_numpy(sc.transform(Xtr).astype(np.float32)); yt=torch.from_numpy(ytr.astype(np.float32)).unsqueeze(1)
    Xv=torch.from_numpy(sc.transform(Xva).astype(np.float32)); yv=torch.from_numpy(yva.astype(np.float32)).unsqueeze(1)
    best=1e9; bestst=None; bad=0
    for ep in range(80):
        mlp.train(); perm=torch.randperm(len(Xt)); tot=0
        for i in range(0,len(Xt),256):
            idx=perm[i:i+256]; opt.zero_grad()
            out=mlp(Xt[idx])[head]; l=lossf(out,yt[idx]); l.backward(); opt.step(); tot+=l.item()
        mlp.eval()
        with torch.no_grad(): vl=lossf(mlp(Xv)[head],yv).item()
        sched.step(vl)
        if vl<best-1e-4: best=vl; bestst={k:v.clone() for k,v in mlp.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=10: break
    mlp.load_state_dict(bestst); mlp.eval()
    return mlp, sc
res={}
for ep,csvp,col,embp in [('PAMPA','base.csv','PAMPA_MDCK','data/molformer/molformer_emb_pampa_mdck.npz'),
                          ('CACO2','caco2.csv','Caco2','data/molformer/molformer_emb_caco2.npz')]:
    df=pd.read_csv(csvp); sm=df['smiles'].astype(str).tolist(); y=df[col].to_numpy(np.float64)
    X=np.hstack([feature_extractor.molecule_features(sm), np.load(embp)['emb'].astype(np.float32)]).astype(np.float32)
    for s in (42,7,123):
        tr,va,te=split(sm,s)
        mlp,sc=train_mlp(X[tr],y[tr],X[va],y[va],X.shape[1],col)
        with torch.no_grad(): p=mlp(torch.from_numpy(sc.transform(X)))['%s'%col].squeeze(1).numpy().astype(np.float64)
        yt=y[te]; pt=p[te]; fl=yt<=-10.0+1e-9
        r2a=r2_score(yt,pt); r2nf=r2_score(yt[~fl],pt[~fl])
        print(f'{ep} split{s} RETRAINED baseline: r2_all={r2a:.4f} r2_nf={r2nf:.4f} floor={int(fl.sum())}',flush=True)
        pf=np.full(len(y),np.nan); pf[te]=pt
        np.savez(f'results/predictions/baseline_retrained_{ep}_split{s}.npz',y=y,tr=tr,va=va,te=te,p=pf)
        res[f'{ep}/split{s}']={'r2_all':float(r2a),'r2_nonfloor':float(r2nf),'n_floor_test':int(fl.sum())}
json.dump(res,open('retrained_baselines.json','w'),indent=1)
