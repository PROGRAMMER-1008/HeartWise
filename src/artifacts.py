"""Trusted local immutable bundles. Checksums detect corruption, not malicious publishers."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from .data_pipeline import ROOT, ALL_FEATURES
from .evaluation import metrics

BUNDLES = ROOT/'artifacts'
DEPENDENCIES = ['scikit-learn','numpy','pandas','joblib','xgboost']

class Manifest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    schema_version: int
    artifact_id: str
    fingerprint: str
    dataset_hash: str
    split_hash: str
    created_utc: str
    python: str
    dependencies: dict[str,str]
    code_revision: str | None
    config: dict
    classes: list[int]
    threshold: float
    calibration: str
    selected: str
    selection_rule: str
    metrics: dict
    uncertainty: dict
    training_support: dict
    checksums: dict[str,str]
    figures: dict[str,str]
    data_quality: dict
    evaluation_scope: str

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def versions():
    return {name:importlib.metadata.version(name) for name in DEPENDENCIES}

def active_path():
    artifact_id=os.getenv('HEARTWISE_ARTIFACT')
    if not artifact_id:
        artifact_id=(BUNDLES/'active.txt').read_text().strip()
    if not re.fullmatch(r'hw-[a-f0-9]{16}',artifact_id):
        raise ValueError('Invalid artifact ID; activate a validated bundle')
    return BUNDLES/artifact_id

def validate_bundle(path, load=True):
    path=Path(path)
    m=Manifest.model_validate_json((path/'manifest.json').read_text()).model_dump()
    if m['schema_version']!=1 or m['classes']!=[0,1] or m['threshold']!=.5:
        raise ValueError('Unsupported bundle schema or decision policy')
    required={'model.joblib','background.csv','test_predictions.csv','validation_predictions.csv','splits.csv','candidates.csv','subgroups.json','nested_cv.json','reliability.csv','thresholds.csv'}
    if not required <= set(m['checksums']):
        raise ValueError('Incomplete bundle')
    for name,digest in m['checksums'].items():
        file=(path/name).resolve()
        if not file.is_relative_to(path.resolve()) or sha(file)!=digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    if not set(m['figures'].values()) <= set(m['checksums']):
        raise ValueError('Unverified figure')
    table=pd.read_csv(path/'test_predictions.csv')
    if not np.isfinite(table.probability).all() or not table.probability.between(0,1).all():
        raise ValueError('Invalid saved probabilities')
    if not np.array_equal(table.predicted,(table.probability>=m['threshold']).astype(int)):
        raise ValueError('Saved classifications disagree with threshold')
    actual=metrics(table.actual,table.probability,m['threshold'])
    if any((v is None and m['metrics'][k] is not None) or (v is not None and not np.isclose(v,m['metrics'][k],atol=1e-12)) for k,v in actual.items()):
        raise ValueError('Metrics do not match saved predictions')
    splits=pd.read_csv(path/'splits.csv')
    if sha(path/'splits.csv')!=m['split_hash'] or splits.row_hash.duplicated().any():
        raise ValueError('Invalid split identity')
    train=set(splits.loc[splits.partition=='development','row_id'])
    test=set(splits.loc[splits.partition=='test','row_id'])
    validation=pd.read_csv(path/'validation_predictions.csv')
    if train & test or set(table.row_id)!=test or set(validation.row_id)!=train:
        raise ValueError('Partition mismatch')
    if load:
        if sys.version_info[:2]!=(3,12) or any(importlib.metadata.version(k)!=v for k,v in m['dependencies'].items()):
            raise ValueError('Incompatible runtime: install Python 3.12 and requirements.lock before serving')
        model=joblib.load(path/'model.joblib')
        if list(model.classes_)!=m['classes']:
            raise ValueError('Model class ordering mismatch')
        background=pd.read_csv(path/'background.csv')[ALL_FEATURES]
        probability=model.predict_proba(background.iloc[:1])[0,1]
        if not np.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Canary inference failed')
        replay=model.predict_proba(table[ALL_FEATURES])[:,1]
        if not np.allclose(replay,table.probability,atol=1e-12):
            raise ValueError('Model does not reproduce saved predictions')
    return m

def activate(path):
    path=Path(path).resolve()
    if path.parent!=BUNDLES.resolve() or not re.fullmatch(r'hw-[a-f0-9]{16}',path.name):
        raise ValueError('Activation requires a versioned local bundle')
    validate_bundle(path)
    temp=BUNDLES/'active.tmp'
    temp.write_text(path.name,encoding='utf-8')
    os.replace(temp,BUNDLES/'active.txt')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['validate','activate'])
    parser.add_argument('bundle',nargs='?')
    args=parser.parse_args()
    path=BUNDLES/args.bundle if args.bundle else active_path()
    if args.action=='activate':
        activate(path)
    else:
        validate_bundle(path)
    print(f'{args.action}: {path.name} OK')

if __name__=='__main__':
    main()
