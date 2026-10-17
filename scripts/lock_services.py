"""Generate service lock subsets from the installed, tested dependency graph."""
from pathlib import Path
from importlib import metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

def closure(roots):
    resolved={}
    def visit(name):
        name=canonicalize_name(name)
        if name in resolved: return
        dist=metadata.distribution(name); resolved[name]=dist.version
        for spec in dist.requires or []:
            req=Requirement(spec)
            if req.marker is None or req.marker.evaluate({'extra':''}): visit(req.name)
    for root in roots: visit(root)
    return '\n'.join(f'{k}=={v}' for k,v in sorted(resolved.items()))+'\n'
base=['scikit-learn','numpy','pandas','joblib','pydantic','requests','reportlab','xgboost']
Path('requirements-api.lock').write_text(closure(base+['fastapi','uvicorn']))
Path('requirements-ui.lock').write_text(closure([x for x in base if x!='xgboost']+['streamlit']))
