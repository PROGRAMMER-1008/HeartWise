"""FastAPI lifecycle, readiness, bounded synthetic-demo serving and privacy-safe telemetry."""
from contextlib import asynccontextmanager
import json
import logging
import os
import time
import uuid
from collections import deque
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from src.explain import get_explainer, DEMO
from src.schemas import PatientInput, PredictionResponse

logger=logging.getLogger('heartwise')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

@asynccontextmanager
async def lifespan(app):
    app.state.explainer=None
    try:
        service=get_explainer()
        service.predict(DEMO,explain=False)
        app.state.explainer=service
    except Exception as exc:
        logger.error('startup_failed error_class=%s',type(exc).__name__)
    yield
    app.state.explainer=None

app=FastAPI(title='HeartWise research API',version='3.0.0',lifespan=lifespan,
            description='Educational Cleveland disease-status classification; not future cardiovascular risk.')
origins=[x.strip() for x in os.getenv('HEARTWISE_CORS_ORIGINS','').split(',') if x.strip()]
if origins:
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])

class RequestLimits:
    def __init__(self,app):
        self.app=app; self.arrivals=deque()
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        request_id=uuid.uuid4().hex; started=time.perf_counter(); status=500
        scope.setdefault('state',{})['request_id']=request_id
        async def tagged_send(message):
            nonlocal status
            if message['type']=='http.response.start':
                status=message['status']
                message['headers'].append((b'x-request-id',request_id.encode()))
            await send(message)
        try:
            if scope['method']=='POST':
                now=time.monotonic(); limit=int(os.getenv('HEARTWISE_REQUESTS_PER_MINUTE','120'))
                while self.arrivals and self.arrivals[0]<now-60:
                    self.arrivals.popleft()
                if len(self.arrivals)>=limit:
                    return await JSONResponse({'detail':'Request limit reached; try later'},status_code=429)(scope,receive,tagged_send)
                self.arrivals.append(now)
                body=b''; maximum=int(os.getenv('HEARTWISE_MAX_BODY_BYTES','8192'))
                while True:
                    message=await receive()
                    if message['type']=='http.disconnect':
                        return
                    body+=message.get('body',b'')
                    if len(body)>maximum:
                        return await JSONResponse({'detail':'Request body too large'},status_code=413)(scope,receive,tagged_send)
                    if not message.get('more_body'):
                        break
                async def replay():
                    return {'type':'http.request','body':body,'more_body':False}
                await self.app(scope,replay,tagged_send)
            else:
                await self.app(scope,receive,tagged_send)
        finally:
            service=getattr(app.state,'explainer',None)
            logger.info(json.dumps({'request_id':request_id,'status':status,'artifact_id':service.metadata['artifact_id'] if service else None,
                                    'duration_ms':round((time.perf_counter()-started)*1000,2)}))

app.add_middleware(RequestLimits)

@app.exception_handler(RequestValidationError)
async def invalid_request(request:Request,exc):
    # Pydantic's default response includes the submitted input; omit it.
    return JSONResponse({'detail':[{'field':'.'.join(map(str,e['loc'])),'type':e['type']} for e in exc.errors()]},status_code=422)

@app.exception_handler(Exception)
async def safe_failure(request:Request,exc):
    logger.error('request_failed error_class=%s',type(exc).__name__)
    return JSONResponse({'detail':'Service error; retry or contact the operator with the request ID'},status_code=500,headers={'X-Request-ID':getattr(request.state,'request_id','unavailable')})

def service():
    result=getattr(app.state,'explainer',None)
    if result is None:
        raise HTTPException(503,'Model unavailable; operator must validate the configured bundle and runtime')
    return result

@app.get('/health')
def health():
    return {'status':'alive','application_version':'3.0.0'}

@app.get('/ready')
def ready():
    engine=service()
    try:
        engine.predict(DEMO,explain=False)
    except Exception:
        raise HTTPException(503,'Inference canary failed') from None
    return {'status':'ready','artifact_id':engine.metadata['artifact_id']}

@app.get('/model-info')
def model_info():
    return service().metadata

@app.get('/evaluation')
def evaluation():
    engine=service(); path=engine.path
    return {'artifact_id':engine.metadata['artifact_id'],
            'candidates':pd.read_csv(path/'candidates.csv').to_dict('records'),
            'thresholds':json.loads(pd.read_csv(path/'thresholds.csv').to_json(orient='records')),
            'reliability':json.loads(pd.read_csv(path/'reliability.csv').to_json(orient='records')),
            'nested_cv':json.loads((path/'nested_cv.json').read_text()),
            'subgroups':json.loads((path/'subgroups.json').read_text())}

@app.post('/predict',response_model=PredictionResponse)
def predict(patient:PatientInput,explain:bool=True):
    engine=service()
    try:
        return PredictionResponse.model_validate(engine.predict(patient.model_dump(),explain=explain))
    except Exception as exc:
        logger.error('prediction_failed error_class=%s',type(exc).__name__)
        raise HTTPException(500,'Prediction failed; contact the operator with the request ID') from None

@app.get('/figures/{filename}')
def figure(filename:str):
    engine=service()
    if filename not in engine.metadata['figures'].values():
        raise HTTPException(404,'Figure unavailable')
    return FileResponse(engine.path/filename,media_type='image/png',headers={'X-Artifact-ID':engine.metadata['artifact_id']})
