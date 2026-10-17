"""Explicit serving modes; API errors never silently fall back."""
import os
import json
from pathlib import Path
import pandas as pd
import requests
from .schemas import PatientInput, PredictionResponse

class ServingClient:
    def __init__(self,mode=None,url=None):
        self.mode=mode or os.getenv('HEARTWISE_MODE','local')
        if self.mode not in {'local','api'}:
            raise ValueError('HEARTWISE_MODE must be local or api')
        self.url=(url or os.getenv('HEARTWISE_API_URL','http://127.0.0.1:8000')).rstrip('/')
        self.timeout=float(os.getenv('HEARTWISE_TIMEOUT_SECONDS','15'))
    def _get(self,path):
        response=requests.get(self.url+path,timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    def info(self):
        if self.mode=='api':
            return self._get('/model-info')
        from .explain import get_explainer
        return get_explainer().metadata
    def predict(self,record,explain=True):
        record=PatientInput.model_validate(record).model_dump()
        if self.mode=='api':
            response=requests.post(self.url+'/predict',params={'explain':explain},json=record,timeout=self.timeout)
            response.raise_for_status()
            return PredictionResponse.model_validate(response.json()).model_dump()
        from .explain import get_explainer
        return get_explainer().predict(record,explain)
    def evaluation(self):
        if self.mode=='api':
            return self._get('/evaluation')
        from .explain import get_explainer
        engine=get_explainer(); path=Path(engine.path)
        return {'artifact_id':engine.metadata['artifact_id'],
                **{key:json.loads(pd.read_csv(path/(key+'.csv')).to_json(orient='records')) for key in ['candidates','thresholds','reliability']},
                **{key:json.loads((path/(key+'.json')).read_text()) for key in ['nested_cv','subgroups']}}

    def figure(self,filename,artifact_id):
        if self.mode=='api':
            response=requests.get(self.url+'/figures/'+filename,timeout=self.timeout)
            response.raise_for_status()
            if response.headers.get('X-Artifact-ID')!=artifact_id:
                raise ValueError('Figure artifact mismatch')
            return response.content
        from .explain import get_explainer
        engine=get_explainer()
        if engine.metadata['artifact_id']!=artifact_id or filename not in engine.metadata['figures'].values():
            raise ValueError('Figure unavailable or artifact changed')
        return (engine.path/filename).read_bytes()
