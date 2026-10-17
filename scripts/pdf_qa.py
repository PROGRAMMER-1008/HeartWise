"""Create synthetic-only PDF QA fixtures and rasterize every page."""
from pathlib import Path
import json
import pymupdf as fitz
from src.explain import get_explainer, DEMO, DEMO_B
from src.state import assessment

engine=get_explainer()
for name,record,explain in [('sample',DEMO,True),('long-unicode',{**DEMO_B,'patient_id':'<b>'+('éΩ&'*18)+'</b>'},True),('unavailable',DEMO,False),('unsupported',{**DEMO,'age':12},False)]:
    snapshot=assessment(record,engine.predict(record,explain))
    Path(f'output/pdf/{name}.pdf').write_bytes(snapshot['pdf'])
    doc=fitz.open(stream=snapshot['pdf'],filetype='pdf')
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(f'tmp/pdfs/{name}-{i+1}.png')
    print(name,len(doc))
Path('docs/sample_request.json').write_text(json.dumps(DEMO,indent=2))
Path('docs/sample_response.json').write_text(json.dumps(engine.predict(DEMO),indent=2))
