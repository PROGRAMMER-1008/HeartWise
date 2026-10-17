"""External compatibility gate: do not invent specialist measurements."""
import json
from src.audit import canonical_uci
from src.artifacts import ROOT, active_path, sha
from src.data_pipeline import ALL_FEATURES
from src.schemas import PatientInput

report={'artifact_id':active_path().name,'source':'https://archive.ics.uci.edu/dataset/45/heart+disease','doi':'10.24432/C52P4X','license':'CC BY 4.0','retrieved_date':'2026-10-12','cohorts':{}}
for name in ['hungarian','switzerland','va']:
    path=ROOT/f'data/reference/processed.{name}.data'; df=canonical_uci(path)
    complete=df.dropna(); valid=[]
    for _,row in complete.iterrows():
        record={c:int(row[c]) if c in ['age','sex','cp','fbs','restecg','exang','slope','ca','thal'] else float(row[c]) for c in ALL_FEATURES}
        try:
            PatientInput.model_validate(record)
            if 29<=record['age']<=77: valid.append(record)
        except ValueError:
            pass
    report['cohorts'][name]={'source_file':path.name,'sha256':sha(path),'n':len(df),'missing_by_field':df.isna().sum().to_dict(),
                            'complete_cases':len(complete),'valid_complete_adult_records':len(valid),
                            'class_counts_complete':complete.target.value_counts().to_dict(),
                            'outcome':'UCI num > 0, same documented disease-status definition',
                            'decision':'No performance estimate: fewer than 20 valid complete records; severe specialist-field missingness makes complete-case selection unrepresentative.' if len(valid)<20 else 'Requires explicit cohort protocol before evaluation'}
report['limitations']='Hospital referral populations differ. No external rows used for selection, calibration or fitting. Missing specialist measurements are not replaced by invented normal values. External accuracy is unestablished.'
(ROOT/'docs/external_cohort_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:{f:v[f] for f in ['n','complete_cases','valid_complete_adult_records']} for k,v in report['cohorts'].items()},indent=2))
