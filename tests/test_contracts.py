"""Focused correctness and serving regression tests, using the validated bundle."""
from pathlib import Path
import copy
import io
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfReader
from src.audit import audit
from src.artifacts import active_path, validate_bundle, activate, sha
from src.data_pipeline import CATEGORY_OPTIONS, load_dataset, ALL_FEATURES, validate_patient_input
from src.explain import get_explainer, DEMO
from src.schemas import PatientInput
from src.state import assessment, is_stale
from src.report import safe_filename

@pytest.fixture(scope='session')
def engine():
    return get_explainer()

def test_source_semantics_and_profile(tmp_path):
    report=audit()
    assert (report['matches'],report['distinct_source_matches'],report['removed_rows'],report['removed_distinct_rows'])==(303,302,7,6)
    assert CATEGORY_OPTIONS['cp'][0]=='Asymptomatic'
    assert CATEGORY_OPTIONS['cp'][3]=='Typical angina'
    assert CATEGORY_OPTIONS['restecg'][1]=='Normal'
    assert CATEGORY_OPTIONS['slope'][2]=='Upsloping'
    assert len(load_dataset())==296
    with pytest.raises(ValueError,match='profile'):
        load_dataset(profile='arbitrary')
    path=tmp_path/'other.csv'; path.write_text('target\n1\n')
    with pytest.raises(ValueError,match='hash'):
        load_dataset(path)

@pytest.mark.parametrize('change',[{'cp':1.9},{'sex':True},{'ca':1.0},{'chol':float('nan')},{'oldpeak':float('inf')},{'age':'54'},{'chol':None},{'unexpected':2}])
def test_invalid_local_api_parity(change):
    from api.main import app
    record={**DEMO,**change}
    assert validate_patient_input(record)
    with pytest.raises(ValidationError):
        PatientInput.model_validate(record)
    with TestClient(app) as client:
        # JSON NaN/Infinity is malformed JSON; exercise the actual boundary.
        response=client.post('/predict',content=json.dumps(record),headers={'content-type':'application/json'})
        assert response.status_code==422
        assert 'input' not in response.json()['detail'][0]

def test_api_readiness_failure_sanitized(monkeypatch):
    import api.main as api
    monkeypatch.setattr(api,'get_explainer',lambda:(_ for _ in ()).throw(RuntimeError('secret/path')))
    with TestClient(api.app) as client:
        assert client.get('/health').status_code==200
        assert client.get('/ready').status_code==503
        assert client.post('/predict',json=DEMO).status_code==503
        assert 'secret' not in client.get('/ready').text

def test_prediction_explanation_and_abstention(engine):
    result=engine.predict(DEMO)
    e=result['explanation']
    assert e['status']=='available' and e['target']==result['artifact_id']
    assert {f['field'] for f in e['contributions']}==set(ALL_FEATURES)
    assert len(e['contributions'])==13
    assert abs(e['base_value']+sum(f['impact'] for f in e['contributions'])-result['probability'])<1e-10
    assert abs(e['residual']-sum(f['impact'] for f in e['contributions'][5:]))<1e-10
    assert next(f for f in e['contributions'] if f['field']=='restecg')['value']=='Normal'
    assert engine.predict({**DEMO,'age':12})['domain_status']=='abstained'
    assert engine.predict({**DEMO,'age':12})['probability'] is None
    assert engine.predict({**DEMO,'chol':600})['warnings']

def test_threshold_and_explanation_failure(engine,monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(engine.model,'predict_proba',lambda X:np.tile([.5,.5],(len(X),1)))
        assert engine.predict(DEMO,False)['prediction']==1
        patch.setattr(engine,'explanation',lambda *a:(_ for _ in ()).throw(ValueError('failure')))
        result=engine.predict(DEMO)
        assert result['probability']==.5 and result['explanation']['status']=='unavailable'

def test_bundle_consistency_and_bad_activation(tmp_path):
    path=active_path(); m=validate_bundle(path)
    assert sum(m['metrics'][k] for k in ['tn','fp','fn','tp'])==m['metrics']['n']
    clone=tmp_path/path.name; shutil.copytree(path,clone)
    (clone/'model.joblib').write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):
        validate_bundle(clone)
    with pytest.raises(ValueError):
        activate(clone)
    assert active_path()==path
    assert sha(path/'splits.csv')==m['split_hash']

def test_nested_partitions_and_train_only_selection(monkeypatch):
    import src.train as training
    path=active_path(); splits=pd.read_csv(path/'splits.csv')
    test=set(splits.loc[splits.partition=='test','row_id'])
    nested=json.loads((path/'nested_cv.json').read_text())
    for fold in nested:
        tr,va=set(fold['train_row_ids']),set(fold['validation_row_ids'])
        assert not tr&va and not (tr|va)&test
    data=load_dataset().sample(48,random_state=42); fits=[]
    class Spy:
        def fit(self,X,y):
            self.ids=set(X.index); fits.append(self.ids); return self
        def predict_proba(self,X):
            assert not self.ids&set(X.index)
            return np.tile([.5,.5],(len(X),1))
    monkeypatch.setattr(training,'candidates',lambda:([('spy','none')],{}))
    monkeypatch.setattr(training,'procedure',lambda *a:Spy())
    training.select_procedure(data[ALL_FEATURES],data.target,3)
    assert len(fits)==3

def test_fold_preprocessing_smoke():
    from src.train import procedure
    data=load_dataset().sample(60,random_state=42)
    calibrated=procedure('Logistic C=0.1','sigmoid').fit(data[ALL_FEATURES],data.target)
    assert list(calibrated.classes_)==[0,1]
    assert calibrated.ensemble is False
    assert np.isfinite(calibrated.predict_proba(pd.DataFrame([DEMO])[ALL_FEATURES])).all()

def test_remote_mode_no_fallback(monkeypatch):
    import requests
    from src.client import ServingClient
    import src.explain
    monkeypatch.setattr(src.explain,'get_explainer',lambda:pytest.fail('Local model used in API mode'))
    class Response:
        def raise_for_status(self):
            raise requests.HTTPError('422')
    monkeypatch.setattr(requests,'post',lambda *a,**k:Response())
    with pytest.raises(requests.HTTPError):
        ServingClient('api').predict(DEMO)

def test_pdf_snapshot_escaping_and_fidelity(engine):
    record={**DEMO,'patient_id':'<b>é & Ω 漢</b>'}
    snapshot=assessment(record,engine.predict(record)); original=snapshot['pdf']
    assert is_stale({**record,'age':55},snapshot)
    record['age']=55
    assert snapshot['inputs']['age']==54 and snapshot['pdf']==original
    reader=PdfReader(io.BytesIO(original)); text='\n'.join(p.extract_text() for p in reader.pages)
    assert '<b>é & Ω [U+6F22]</b>' in text
    assert snapshot['artifact_id'] in text and 'Normal' in text
    assert len(reader.pages)>=2 and all('Page' in p.extract_text() for p in reader.pages)
    assert safe_filename('../../bad<id>')=='heartwise_bad_id.pdf'

def test_api_response_limits_and_identity(engine,monkeypatch):
    from api.main import app
    with TestClient(app) as client:
        response=client.post('/predict',json=DEMO)
        assert response.status_code==200 and response.headers['x-request-id']
        assert response.json()['artifact_id']==client.get('/model-info').json()['artifact_id']
        assert client.get('/evaluation').status_code==200
        figure=client.get('/figures/roc.png')
        assert figure.status_code==200 and figure.headers['x-artifact-id']==engine.metadata['artifact_id']
        assert client.get('/figures/manifest.json').status_code==404
        assert client.post('/predict',content=b'x'*9000).status_code==413
        assert client.get('/ready').status_code==200
        assert 'access-control-allow-origin' not in response.headers

def test_streamlit_snapshot_rerun_and_clear():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(Path('app/streamlit_app.py').resolve(),default_timeout=20).run()
    assert not at.exception
    at.checkbox(key='review_confirmed').check().run()
    next(b for b in at.button if b.label=='Assess record').click().run()
    assert not at.exception
    before=copy.deepcopy(at.session_state['assessment'])
    at.radio[0].set_value('Model evidence').run()
    assert not at.exception
    at.radio[0].set_value('Assessment').run()
    assert at.number_input(key='input_age').value == 54
    assert at.session_state['assessment']['pdf']==before['pdf']
    at.number_input(key='input_age').set_value(55).run()
    assert any('Inputs have changed' in x.value for x in at.warning)
    assert not at.checkbox(key='review_confirmed').value
    next(b for b in at.button if b.label=='Clear session').click().run()
    assert 'assessment' not in at.session_state
    assert not at.exception

def test_atomic_activation_and_rollback(tmp_path,monkeypatch):
    import src.artifacts as artifacts
    from src.train import figures
    production=active_path()
    root=tmp_path/'bundles'; root.mkdir()
    releases=[]
    for identity in ['hw-1111111111111111','hw-2222222222222222']:
        dest=root/identity; shutil.copytree(production,dest)
        manifest=json.loads((dest/'manifest.json').read_text())
        manifest['artifact_id']=identity
        figures(dest,pd.read_csv(dest/'test_predictions.csv'),identity)
        manifest['checksums']={name:sha(dest/name) for name in manifest['checksums']}
        (dest/'manifest.json').write_text(json.dumps(manifest))
        releases.append(dest)
    monkeypatch.setattr(artifacts,'BUNDLES',root)
    monkeypatch.delenv('HEARTWISE_ARTIFACT',raising=False)
    artifacts.activate(releases[0]); assert artifacts.active_path()==releases[0]
    artifacts.activate(releases[1]); assert artifacts.active_path()==releases[1]
    artifacts.activate(releases[0]); assert artifacts.active_path()==releases[0]
    (releases[1]/'model.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        artifacts.activate(releases[1])
    assert artifacts.active_path()==releases[0]

def test_api_failure_request_id_and_no_payload(engine,monkeypatch):
    from api.main import app
    with TestClient(app,raise_server_exceptions=False) as client:
        monkeypatch.setattr(engine,'predict',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('PRIVATE_PATIENT_SECRET')))
        response=client.post('/predict',json=DEMO)
        assert response.status_code==500 and response.headers['x-request-id']!='unavailable'
        assert 'PRIVATE_PATIENT_SECRET' not in response.text

def test_imputation_study_keeps_final_test_out():
    from src.data_pipeline import ROOT
    study=json.loads((ROOT/'docs/imputation_study.json').read_text())
    path=ROOT/study['study_path']
    manifest=json.loads((path/'manifest.json').read_text())
    split=pd.read_csv(active_path()/'splits.csv')
    final_ids=set(split.loc[split.partition=='test','row_id'])
    for fold in manifest['folds']:
        train,evaluated=set(fold['training_rows']),set(fold['evaluated_rows'])
        assert not train&evaluated and not (train|evaluated)&final_ids
    table=pd.read_csv(path/'development_predictions.csv')
    assert table.groupby('method').row_id.nunique().to_dict()=={'complete_case':236,'fold_imputation':236}
    assert sha(path/'development_predictions.csv')==manifest['checksums']['development_predictions.csv']
