"""Diagnostic development-only missing-data comparison; never activates a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.artifacts import active_path, sha
from src.data_pipeline import ROOT, DATA_PATH, DATA_HASH, ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CATEGORY_OPTIONS
from src.evaluation import metrics


def estimator(impute):
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]) if impute else StandardScaler()
    encoder=OneHotEncoder(categories=[sorted(CATEGORY_OPTIONS[c]) for c in CATEGORICAL_FEATURES],handle_unknown='error')
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encode',encoder)]) if impute else encoder
    prep=ColumnTransformer([('num',numeric,NUMERIC_FEATURES),('cat',categorical,CATEGORICAL_FEATURES)])
    base=Pipeline([('preprocessor',prep),('classifier',LogisticRegression(C=.1,max_iter=2000,random_state=42))])
    return CalibratedClassifierCV(base,method='sigmoid',ensemble=False,cv=StratifiedKFold(3,shuffle=True,random_state=42))


def main():
    bundle=active_path(); split=pd.read_csv(bundle/'splits.csv')
    excluded=set(split.loc[split.partition=='test','row_id'])
    if sha(DATA_PATH)!=DATA_HASH:
        raise ValueError('Audit required before comparison')
    raw=pd.read_csv(DATA_PATH)
    # Remove final test rows BEFORE constructing development X/y. Deduplicate missing-data records.
    dev=raw.loc[~raw.index.isin(excluded)].drop_duplicates().copy()
    dev.target=1-dev.target
    dev.loc[dev.ca==4,'ca']=np.nan; dev.loc[dev.thal==0,'thal']=np.nan
    config={'artifact_id':bundle.name,'dataset_hash':DATA_HASH,'split_hash':sha(bundle/'splits.csv'),
            'model':'fixed logistic C=.1 + sigmoid ensemble=False','seed':42,'outer_folds':3,
            'comparison':'train complete-case removal vs fold-local median/mode imputation; evaluate common complete validation rows',
            'code_hash':sha(Path(__file__))}
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    path=ROOT/'artifacts/studies'/('imputation-'+identity)
    if (path/'manifest.json').exists():
        print('Reused diagnostic study '+path.name); return
    path.mkdir(parents=True,exist_ok=False)
    results=[]; isolation=[]
    X,y=dev[ALL_FEATURES],dev.target
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(X,y)):
        complete_validation=X.iloc[va].dropna()
        isolation.append({'fold':fold,'training_rows':X.iloc[tr].index.tolist(),'evaluated_rows':complete_validation.index.tolist()})
        assert not (set(X.iloc[tr].index)|set(complete_validation.index)) & excluded
        for impute in [False,True]:
            training=X.iloc[tr] if impute else X.iloc[tr].dropna()
            fitted=estimator(impute).fit(training,y.loc[training.index])
            probability=fitted.predict_proba(complete_validation)[:,1]
            results.extend({'method':'fold_imputation' if impute else 'complete_case','fold':fold,'row_id':int(idx),
                            'actual':int(y.loc[idx]),'probability':float(p)} for idx,p in zip(complete_validation.index,probability))
    table=pd.DataFrame(results); table.to_csv(path/'development_predictions.csv',index=False)
    report={'config':config,'development_rows_including_missing':len(dev),'missing_records':int(X.isna().any(axis=1).sum()),
            'common_evaluation_rows':int((~X.isna().any(axis=1)).sum()),'folds':isolation,
            'metrics':{name:metrics(g.actual,g.probability) for name,g in table.groupby('method')},
            'limitation':'Diagnostic only, on development data already used for selection; not a new unbiased estimate. Evaluation excludes incomplete validation records, so this does not validate scoring missing specialist inputs. Deployed model and complete-input policy remain unchanged.',
            'checksums':{'development_predictions.csv':sha(path/'development_predictions.csv')}}
    (path/'manifest.json').write_text(json.dumps(report,indent=2))
    (ROOT/'docs/imputation_study.json').write_text(json.dumps({'study_path':str(path.relative_to(ROOT)),'metrics':report['metrics'],'limitation':report['limitation']},indent=2))
    print(json.dumps({'study':path.name,'n':len(dev),'metrics':report['metrics']},indent=2))

if __name__=='__main__':
    main()
