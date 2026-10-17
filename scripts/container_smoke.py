"""Run inside the API-only UI image; no local artifacts may exist."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import time
import requests
from src.client import ServingClient
from src.explain import DEMO
from src.state import assessment

assert not Path('/app/artifacts').exists()
client=ServingClient(); assert client.mode=='api'
ready=requests.get(client.url+'/ready',timeout=10); assert ready.status_code==200
info=client.info(); result=client.predict(DEMO)
assert result['artifact_id']==info['artifact_id']==ready.json()['artifact_id']
assert result['explanation']['status']=='available'
assert len(assessment(DEMO,result)['pdf'])>1000
assert client.evaluation()['artifact_id']==info['artifact_id']
assert client.figure('roc.png',info['artifact_id']).startswith(b'\x89PNG')
bad=requests.post(client.url+'/predict',json={**DEMO,'cp':1.8},timeout=10)
assert bad.status_code==422
assert requests.post(client.url+'/predict',data=b'x'*9000,timeout=10).status_code==413
start=time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as pool:
    predictions=list(pool.map(lambda _:client.predict(DEMO),range(8)))
assert all(r['probability']==result['probability'] and r['artifact_id']==result['artifact_id'] for r in predictions)
print(json.dumps({'artifact_id':result['artifact_id'],'ui_has_local_artifacts':False,'concurrent_requests':8,'workers':4,
                  'total_concurrent_seconds':time.perf_counter()-start,'status':'passed'}))
