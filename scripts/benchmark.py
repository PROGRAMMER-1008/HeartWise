"""Small synthetic workload, no clinical payloads in output."""
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from src.explain import HeartRiskExplainer, DEMO

start=time.perf_counter(); engine=HeartRiskExplainer(); startup=time.perf_counter()-start
engine.predict(DEMO)
def summary(values):
    return {'n':len(values),'p50_ms':float(np.percentile(values,50)*1000),'p95_ms':float(np.percentile(values,95)*1000)}
def timed(explain):
    begin=time.perf_counter(); r=engine.predict(DEMO,explain); return time.perf_counter()-begin,r['probability']
results={}
for flag in [False,True]:
    samples=[timed(flag)[0] for _ in range(20)]
    results['with_explanation' if flag else 'prediction_only']=summary(samples)
with ThreadPoolExecutor(max_workers=4) as pool:
    concurrent=list(pool.map(timed,[True]*12))
assert len(set(p for _,p in concurrent))==1
results['concurrent_4_workers']=summary([t for t,_ in concurrent])
# Process working set on Windows; resource peak RSS on Unix.
if sys.platform=='win32':
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    counter=Counters(); counter.cb=ctypes.sizeof(counter)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counter),counter.cb)
    memory={'working_set_mb':counter.WorkingSetSize/1024**2,'peak_working_set_mb':counter.PeakWorkingSetSize/1024**2}
else:
    import resource
    memory={'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
cold=time.perf_counter()
subprocess.run([sys.executable,'-c','from src.explain import get_explainer,DEMO; get_explainer().predict(DEMO,False)'],check=True,capture_output=True)
cold=time.perf_counter()-cold
report={'artifact_id':engine.metadata['artifact_id'],'environment':platform.platform(),'python':platform.python_version(),
        'workload':'Synthetic A; 20 sequential requests per mode, 12 requests/4 concurrent workers; lock serializes estimator use',
        'cold_process_to_prediction_seconds':cold,'bundle_load_seconds':startup,'latency':results,'memory':memory,
        'targets':'Local demo target: startup <5s; warm explained p95 <500ms, 4-worker p95 <2s. Baseline targets, not SLAs.'}
Path('docs/performance.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
