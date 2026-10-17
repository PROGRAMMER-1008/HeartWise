"""Metrics derived solely from a saved prediction table and declared threshold."""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score

def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    both = len(np.unique(y))==2
    return {'n':len(y),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'accuracy':float(accuracy_score(y,pred)), 'precision':float(precision_score(y,pred,zero_division=0)),
            'sensitivity':float(recall_score(y,pred,zero_division=0)) if tp+fn else None,
            'specificity':float(tn/(tn+fp)) if tn+fp else None,
            'roc_auc':float(roc_auc_score(y,p)) if both else None,
            'average_precision':float(average_precision_score(y,p)) if both else None,
            'brier':float(brier_score_loss(y,p))}

def intervals(y,p,seed=42,n=500):
    rng=np.random.default_rng(seed)
    draws={k:[] for k in ['accuracy','sensitivity','specificity','roc_auc','brier']}
    y,p=np.asarray(y),np.asarray(p)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        values=metrics(y[idx],p[idx])
        for key in draws:
            if values[key] is not None:
                draws[key].append(values[key])
    return {'method':'95% percentile bootstrap, fixed fitted model; not training/selection uncertainty',
            'seed':seed,'resamples':n,'limits':{k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items() if v}}

def reliability(y,p):
    y,p=np.asarray(y),np.asarray(p)
    return [{'lower':i/5,'upper':(i+1)/5,'n':int(mask.sum()),
             'mean_probability':float(p[mask].mean()) if mask.any() else None,
             'positive_fraction':float(y[mask].mean()) if mask.any() else None}
            for i in range(5) for mask in [(p>=i/5)&((p<(i+1)/5) if i<4 else (p<=1))]]
