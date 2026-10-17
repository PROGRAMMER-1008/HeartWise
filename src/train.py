"""Bounded training-only selection, nested development estimates, one final test evaluation."""
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from .artifacts import BUNDLES, sha, versions, validate_bundle, activate
from .audit import audit
from .data_pipeline import ALL_FEATURES, NUMERIC_FEATURES, ROOT, load_dataset, make_pipeline
from .evaluation import metrics, intervals, reliability

CONFIG={'seed':42,'test_fraction':.2,'inner_folds':3,'selection_folds':5,'outer_folds':3,
        'calibration':['none','sigmoid'],'calibration_ensemble':False,'threshold':.5,
        'monotonic_policy':'none; correlations are not sufficient justification',
        'missing_policy':'complete cases; imputation comparison deferred to a separately versioned study'}

def candidates():
    models={f'Logistic C={c}':LogisticRegression(C=c,max_iter=2000,random_state=42) for c in [.1,1,10]}
    models.update({'Decision tree':DecisionTreeClassifier(max_depth=3,min_samples_leaf=10,random_state=42),
                   'Random forest':RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_leaf=5,random_state=42,n_jobs=1),
                   'XGBoost':XGBClassifier(n_estimators=80,max_depth=2,learning_rate=.05,subsample=1,colsample_bytree=1,n_jobs=1,random_state=42),
                   'SVM':SVC(C=1,probability=True,random_state=42),
                   'KNN':KNeighborsClassifier(n_neighbors=15)})
    return [(name,cal) for name in models for cal in CONFIG['calibration']],models

def procedure(name,cal):
    _,models=candidates()
    pipe=make_pipeline(models[name])
    if cal=='none':
        return pipe
    return CalibratedClassifierCV(pipe,method=cal,cv=StratifiedKFold(3,shuffle=True,random_state=42),ensemble=False,n_jobs=1)

def select_procedure(X,y,folds=5):
    """Only development X/y enter this function. Every preprocessor/calibrator fits inside fold."""
    rows,predictions=[],{}
    for name,cal in candidates()[0]:
        p=np.zeros(len(y)); fold_scores=[]
        for train_idx,val_idx in StratifiedKFold(folds,shuffle=True,random_state=42).split(X,y):
            estimator=procedure(name,cal).fit(X.iloc[train_idx],y.iloc[train_idx])
            p[val_idx]=estimator.predict_proba(X.iloc[val_idx])[:,1]
            fold_scores.append(metrics(y.iloc[val_idx],p[val_idx]))
        values=metrics(y,p)
        rows.append({'name':name,'calibration':cal,**values,
                     'fold_auc_sd':float(np.std([m['roc_auc'] for m in fold_scores]))})
        predictions[(name,cal)]=p
    # Predeclared probability objective. Lexical identity gives a deterministic exact tie break.
    ranked=sorted(rows,key=lambda r:(r['brier'],-r['roc_auc'],r['name'],r['calibration']))
    winner=ranked[0]
    return winner,pd.DataFrame(rows),predictions[(winner['name'],winner['calibration'])]

def subgroup_report(table):
    groups={'sex=0':table.sex==0,'sex=1':table.sex==1,'age<55':table.age<55,'age>=55':table.age>=55}
    report={}
    for name,mask in groups.items():
        subset=table[mask]
        n0=int((subset.actual==0).sum()); n1=int((subset.actual==1).sum())
        stable=n0>=10 and n1>=10
        report[name]={'n':len(subset),'negative':n0,'positive':n1,'status':'descriptive' if stable else 'unstable; rates suppressed (class denominator <10)',
                      'metrics':metrics(subset.actual,subset.probability) if stable else None,
                      'uncertainty':intervals(subset.actual,subset.probability,n=200) if stable else None,
                      'reliability':reliability(subset.actual,subset.probability) if stable else None}
    return report

def figures(path,table,run_id):
    m=metrics(table.actual,table.probability)
    specs={}
    fig,ax=plt.subplots(figsize=(5,4)); matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
    ax.imshow(matrix,cmap='Reds'); ax.set(xlabel='Predicted class',ylabel='Actual class',xticks=[0,1],yticks=[0,1])
    for (i,j),value in np.ndenumerate(matrix):
        ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black',fontsize=18)
    specs['Confusion matrix']='confusion.png'
    ax.set_title(f'Final test | {run_id}\nThreshold 0.50'); fig.tight_layout(); fig.savefig(path/'confusion.png',dpi=140); plt.close(fig)
    fpr,tpr,_=roc_curve(table.actual,table.probability)
    precision,recall,_=precision_recall_curve(table.actual,table.probability)
    rel=pd.DataFrame(reliability(table.actual,table.probability)); rel.to_csv(path/'reliability.csv',index=False)
    for title,x,y,xlabel,ylabel,file in [
        ('ROC',fpr,tpr,'False positive rate','Sensitivity','roc.png'),
        ('Precision-recall',recall,precision,'Recall','Precision','pr.png'),
        ('Reliability',rel.mean_probability,rel.positive_fraction,'Mean probability','Observed positive fraction','calibration.png')]:
        fig,ax=plt.subplots(figsize=(5,4)); ax.plot(x,y,'o-',color='#a3223a')
        if file=='calibration.png':
            ax.plot([0,1],[0,1],'--',color='gray')
            for row in rel.dropna().itertuples():
                ax.annotate(f'n={row.n}',(row.mean_probability,row.positive_fraction))
        ax.set(xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1),title=f'{title} | {run_id}\nFinal retrospective test')
        fig.tight_layout(); fig.savefig(path/file,dpi=140); plt.close(fig); specs[title]=file
    return specs

def main():
    quality=audit(); df=load_dataset(); X,y=df[ALL_FEATURES],df.target
    code_hash=hashlib.sha256(b''.join((ROOT/'src'/f).read_bytes() for f in ['train.py','evaluation.py','data_pipeline.py','audit.py'])).hexdigest()
    fingerprint=hashlib.sha256(json.dumps({'data':quality['local_sha256'],'config':CONFIG,'versions':versions(),'code':code_hash},sort_keys=True).encode()).hexdigest()
    run_id='hw-'+fingerprint[:16]; destination=BUNDLES/run_id
    BUNDLES.mkdir(exist_ok=True)
    if destination.exists():
        validate_bundle(destination); activate(destination); print(f'Reused validated {run_id}'); return
    staging=BUNDLES/(run_id+'.staging')
    if staging.exists():
        raise RuntimeError(f'Incomplete staging exists at {staging}; inspect it before retrying')
    staging.mkdir()
    dev_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
    Xdev,ydev=X.iloc[dev_idx],y.iloc[dev_idx]
    nested=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=43).split(Xdev,ydev)):
        win,_,_=select_procedure(Xdev.iloc[tr],ydev.iloc[tr],folds=3)
        fitted=procedure(win['name'],win['calibration']).fit(Xdev.iloc[tr],ydev.iloc[tr])
        p=fitted.predict_proba(Xdev.iloc[va])[:,1]
        nested.append({'fold':fold,'selected':win['name'],'calibration':win['calibration'],
                       'train_row_ids':Xdev.iloc[tr].index.tolist(),'validation_row_ids':Xdev.iloc[va].index.tolist(),
                       'metrics':metrics(ydev.iloc[va],p)})
        print(f'Nested development fold {fold+1}/3 complete',flush=True)
    winner,comparison,validation_p=select_procedure(Xdev,ydev)
    comparison.to_csv(staging/'candidates.csv',index=False)
    (staging/'nested_cv.json').write_text(json.dumps(nested,indent=2))
    # Selection is locked before accessing final test outcomes.
    model=procedure(winner['name'],winner['calibration']).fit(Xdev,ydev)
    p=model.predict_proba(X.iloc[test_idx])[:,1]
    table=X.iloc[test_idx].copy(); table['row_id']=table.index; table['actual']=y.iloc[test_idx]; table['probability']=p
    table['predicted']=(p>=.5).astype(int); table.to_csv(staging/'test_predictions.csv',index=False)
    validation=pd.DataFrame({'row_id':Xdev.index,'actual':ydev.to_numpy(),'probability':validation_p})
    validation.to_csv(staging/'validation_predictions.csv',index=False)
    pd.DataFrame([{'threshold':float(t),**metrics(ydev,validation_p,t)} for t in np.linspace(0,1,101)]).to_csv(staging/'thresholds.csv',index=False)
    rows=[]
    for pos,(idx,row) in enumerate(df.iterrows()):
        rows.append({'row_id':idx,'row_hash':hashlib.sha256(row.to_json().encode()).hexdigest(),
                     'source_row_id':quality['local_to_source_rows'][idx], 'partition':'test' if pos in test_idx else 'development'})
    pd.DataFrame(rows).to_csv(staging/'splits.csv',index=False)
    Xdev.sample(8,random_state=42).to_csv(staging/'background.csv',index=False)
    joblib.dump(model,staging/'model.joblib')
    table=pd.read_csv(staging/'test_predictions.csv')
    plots=figures(staging,table,run_id)
    (staging/'subgroups.json').write_text(json.dumps(subgroup_report(table),indent=2))
    manifest={'schema_version':1,'artifact_id':run_id,'fingerprint':fingerprint,'dataset_hash':quality['local_sha256'],
              'split_hash':sha(staging/'splits.csv'),'created_utc':datetime.now(timezone.utc).isoformat(),
              'python':platform.python_version(),'dependencies':versions(),'code_revision':None,
              'config':{**CONFIG,'code_sha256':code_hash},'classes':[0,1],'threshold':.5,
              'calibration':winner['calibration'],'selected':winner['name'],
              'selection_rule':'Lowest development OOF Brier; then highest ROC-AUC; then lexical identity. No final-test selection.',
              'metrics':metrics(table.actual,table.probability),'uncertainty':intervals(table.actual,table.probability),
              'training_support':{c:[float(Xdev[c].min()),float(Xdev[c].max())] for c in NUMERIC_FEATURES},
              'checksums':{f.name:sha(f) for f in staging.iterdir() if f.is_file()},'figures':plots,
              'data_quality':{k:quality[k] for k in ['raw_rows','clean_rows','removed_rows','removed_distinct_rows','matches','distinct_source_matches']},
              'evaluation_scope':'Internal retrospective evaluation on already inspected Cleveland data; not external or clinical validation.'}
    (staging/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validate_bundle(staging)
    os.replace(staging,destination)
    activate(destination)
    print(json.dumps({'artifact_id':run_id,'selected':winner['name'],'calibration':winner['calibration'],'metrics':manifest['metrics']},indent=2))

if __name__=='__main__':
    main()
