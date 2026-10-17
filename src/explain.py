"""Raw-field permutation Shapley estimates of the actual served probability.

Each path replaces complete raw fields (all encoded categories implicitly included).
Averaging paths estimates interventional Shapley values; each path telescopes exactly.
This is a finite-background, finite-permutation association explanation, not causal.
"""
from functools import lru_cache
from threading import RLock
import logging
import numpy as np
import pandas as pd
import joblib
from .artifacts import active_path, validate_bundle
from .data_pipeline import ALL_FEATURES, CATEGORY_OPTIONS, FEATURE_LABELS
from .schemas import PatientInput, PredictionResponse, domain_warnings

DEMO={'age':54,'sex':0,'cp':2,'trestbps':125,'chol':240,'fbs':0,'restecg':1,
      'thalach':155,'exang':0,'oldpeak':.6,'slope':2,'ca':0,'thal':2,'patient_id':'Synthetic A'}
DEMO_B={**DEMO,'age':62,'cp':0,'exang':1,'oldpeak':2.2,'ca':2,'patient_id':'Synthetic B'}
NOTE='Finite-background permutation Shapley estimate of this model probability; associations, not causes or treatment advice. Correlated-feature combinations may be unrealistic.'

class HeartRiskExplainer:
    def __init__(self,path=None):
        self.path=path or active_path()
        self.metadata=validate_bundle(self.path)
        self.model=joblib.load(self.path/'model.joblib')
        self.background=pd.read_csv(self.path/'background.csv')[ALL_FEATURES]
        self.lock=RLock()

    def explanation(self,record,probability,permutations=16):
        rng=np.random.default_rng(42)
        target=np.array([record[c] for c in ALL_FEATURES],dtype=float)
        paths=[]; orders=[]
        for _ in range(permutations):
            order=rng.permutation(len(ALL_FEATURES)); orders.append(order)
            state=self.background.to_numpy().copy(); paths.append(state.copy())
            for j in order:
                state[:,j]=target[j]; paths.append(state.copy())
        batch=pd.DataFrame(np.concatenate(paths),columns=ALL_FEATURES)
        scores=self.model.predict_proba(batch)[:,1].reshape(permutations,len(ALL_FEATURES)+1,len(self.background)).mean(axis=2)
        grouped=np.zeros(len(ALL_FEATURES))
        for order,values in zip(orders,scores):
            grouped[order]+=np.diff(values)/permutations
        base=float(scores[:,0].mean())
        error=abs(base+grouped.sum()-probability)
        if error>1e-8:
            raise ValueError('Explanation reconstruction failed')
        factors=[{'field':c,'feature':FEATURE_LABELS[c],
                  'value':str(CATEGORY_OPTIONS[c][record[c]]) if c in CATEGORY_OPTIONS else str(record[c]),
                  'impact':float(grouped[i])} for i,c in enumerate(ALL_FEATURES)]
        factors.sort(key=lambda f:abs(f['impact']),reverse=True)
        return {'status':'available','target':self.metadata['artifact_id'],'scale':'probability',
                'method':'grouped raw-field permutation Shapley; 16 permutations, 8 development backgrounds, seed 42',
                'base_value':base,'contributions':factors,'residual':sum(f['impact'] for f in factors[5:]),
                'reconstruction_error':float(error),'note':NOTE}

    def predict(self,patient,explain=True):
        record=PatientInput.model_validate(patient).model_dump()
        m=self.metadata; warnings=domain_warnings(record,m['training_support'])
        abstained=not 29<=record['age']<=77
        explanation={'status':'unavailable','target':m['artifact_id'],'method':'not requested or unsupported domain','note':NOTE}
        p=None
        with self.lock:
            if not abstained:
                p=float(self.model.predict_proba(pd.DataFrame([{c:record[c] for c in ALL_FEATURES}]))[0,1])
                if explain:
                    try:
                        explanation=self.explanation(record,p)
                    except Exception as exc:
                        logging.getLogger(__name__).warning('explanation_unavailable error_class=%s artifact=%s',type(exc).__name__,m['artifact_id'])
                        explanation['method']='Explanation failed; prediction remains available'
        result={'artifact_id':m['artifact_id'],'probability':p,'prediction':int(p>=m['threshold']) if p is not None else None,
                'prediction_label':('Disease-present class' if p>=m['threshold'] else 'Disease-absent class') if p is not None else 'Unsupported age: score withheld',
                'threshold':m['threshold'],'calibration':m['calibration'],
                'domain_status':'abstained' if abstained else ('outside_observed_ranges' if warnings else 'within_observed_ranges'),
                'warnings':warnings,'explanation':explanation}
        return PredictionResponse.model_validate(result).model_dump()

@lru_cache(maxsize=2)
def load_explainer(path):
    return HeartRiskExplainer(path)

def get_explainer():
    return load_explainer(active_path())
