"""Session snapshot helpers. No global patient caches or persistent patient files."""
from copy import deepcopy
from datetime import datetime, timezone
from .schemas import PatientInput
from .report import build_pdf_report

def assessment(record,result):
    inputs=PatientInput.model_validate(record).model_dump()
    snapshot={'inputs':deepcopy(inputs),'result':deepcopy(result),'artifact_id':result['artifact_id'],
              'assessed_at':datetime.now(timezone.utc).isoformat()}
    snapshot['pdf']=build_pdf_report(snapshot)
    return snapshot

def is_stale(record,snapshot):
    return snapshot is not None and record!=snapshot['inputs']
