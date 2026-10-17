"""Guided research interface with durable, artifact-bound assessment snapshots."""
import pandas as pd
import streamlit as st
from src.client import ServingClient
from src.data_pipeline import ALL_FEATURES, CATEGORY_OPTIONS, FEATURE_LABELS, NUMERIC_RANGES
from src.explain import DEMO, DEMO_B
from src.schemas import PatientInput, UNITS
from src.state import assessment, is_stale
from src.report import safe_filename

st.set_page_config(page_title='HeartWise | Research classifier',page_icon='❤',layout='wide')
st.html('<style>.block-container{max-width:1150px} h1{color:#c0293b!important} [data-testid="stCaptionContainer"]{opacity:1!important} button:focus-visible{outline:3px solid #225da8!important}</style>')
st.title('HeartWise')
st.write('Explore an explainable disease-status classifier')
st.caption('Educational Cleveland dataset demonstration. Not a diagnosis or a prediction of future heart attacks.')
# Retain widget values when switching workspaces; Streamlit otherwise cleans hidden widgets.
for key in list(st.session_state):
    if key.startswith('input_') or key == 'record_source':
        st.session_state[key] = st.session_state[key]
client=ServingClient()
st.caption(f'Serving mode: {client.mode}')
if st.button('Clear session',help='Remove all inputs, results and generated reports from this session'):
    for key in list(st.session_state):
        del st.session_state[key]
    st.rerun()
try:
    info=client.info()
except Exception:
    st.error('Model service unavailable. Check the configured API or validate the local artifact and Python environment, then retry.')
    st.stop()
st.caption(f"Artifact {info['artifact_id']} · {info['selected']} · Calibration: {info['calibration']}")
page=st.radio('Workspace',['Assessment','Model evidence','Research playground'],horizontal=True)

if page=='Assessment':
    source=st.selectbox('Record source',['Synthetic A','Synthetic B','Complete research record'],key='record_source')
    if st.session_state.get('_source')!=source:
        values=DEMO if source=='Synthetic A' else DEMO_B
        for field in ALL_FEATURES:
            st.session_state['input_'+field]=None if source=='Complete research record' else (float(values[field]) if field in NUMERIC_RANGES and field not in ['age','ca'] else values[field])
        st.session_state['input_patient_id']='Anonymous' if source=='Complete research record' else values['patient_id']
        st.session_state['_source']=source
        st.session_state.pop('review_confirmed',None)
    if source.startswith('Synthetic'):
        st.info('Synthetic demonstration values. They are not a real patient record.')
    else:
        st.info('Supply every measured field. Unknown specialist results remain blank and block scoring.')
    record={}
    groups={'1 · Basic information':['age','sex','trestbps'],
            '2 · Laboratory and ECG':['chol','fbs','restecg'],
            '3 · Specialist and exercise results':['cp','thalach','exang','oldpeak','slope','ca','thal']}
    for heading,fields in groups.items():
        with st.container(border=True):
            st.subheader(heading)
            columns=st.columns(2)
            for i,field in enumerate(fields):
                with columns[i%2]:
                    label=FEATURE_LABELS[field]+(f' ({UNITS[field]})' if field in UNITS else '')
                    if field in CATEGORY_OPTIONS:
                        record[field]=st.selectbox(label,list(CATEGORY_OPTIONS[field]),index=None,
                                                  format_func=lambda x,f=field:CATEGORY_OPTIONS[f][x],key='input_'+field,
                                                  placeholder='Unknown / choose measured category',help='Source-verified local category encoding.')
                    else:
                        lo,hi,step=NUMERIC_RANGES[field]
                        integer=field in ['age','ca']
                        cast=int if integer else float
                        record[field]=st.number_input(label,min_value=cast(lo),max_value=cast(hi),step=cast(step),value=None,
                                                     key='input_'+field,placeholder='Unknown',help='Observed training support: '+str(info['training_support'][field]))
    record['patient_id']=st.text_input('Optional report ID (avoid real identifiers)',max_chars=64,key='input_patient_id')
    with st.expander('Review the current record',expanded=True):
        st.table(pd.DataFrame([{'Field':FEATURE_LABELS[c],'Value':str(CATEGORY_OPTIONS[c].get(record[c],'Unknown') if c in CATEGORY_OPTIONS else record[c])} for c in ALL_FEATURES]),border='horizontal')
    signature=repr(record)
    if st.session_state.get('_review_signature')!=signature:
        st.session_state['review_confirmed']=False
        st.session_state['_review_signature']=signature
    reviewed=st.checkbox('I reviewed these measured or synthetic inputs',key='review_confirmed')
    snapshot=st.session_state.get('assessment')
    if is_stale(record,snapshot):
        st.warning('Inputs have changed. The result and download below still belong to the previous assessed snapshot. Reassess to update them.')
    if st.button('Assess record',type='primary',disabled=not reviewed):
        try:
            normalized=PatientInput.model_validate(record).model_dump()
            with st.spinner('Scoring and explaining the research record…'):
                result=client.predict(normalized)
                if result['artifact_id']!=info['artifact_id']:
                    raise ValueError('Serving artifact changed; refresh model information')
                st.session_state['assessment']=assessment(normalized,result)
        except Exception:
            st.error('Assessment could not complete. Supply all fields with valid numeric values, or check service availability. The previous assessment is retained.')
    snapshot=st.session_state.get('assessment')
    if snapshot:
        result=snapshot['result']; explanation=result['explanation']
        st.subheader('Assessed result')
        st.caption(f"Snapshot: {snapshot['assessed_at']} · {snapshot['artifact_id']}")
        if snapshot['artifact_id']!=info['artifact_id']:
            st.warning('This saved assessment used an earlier artifact. Reassess to use the current model.')
        st.metric('Model probability of disease-present class','Withheld' if result['probability'] is None else f"{result['probability']:.1%}")
        st.write(result['prediction_label'])
        st.caption(f"UCI num > 0 · threshold {result['threshold']:.2f} · calibration {result['calibration']}. This is not a clinical risk band. Individual score uncertainty is not quantified.")
        for warning in result['warnings']:
            st.warning(warning)
        st.caption('Being inside observed ranges does not establish support for the whole profile.')
        if explanation['status']=='available':
            st.subheader('What influenced this model score?')
            st.caption(explanation['note'])
            factors=pd.DataFrame(explanation['contributions'])
            factors['percentage points']=(factors.impact*100).round(2)
            st.bar_chart(factors.set_index('feature')['percentage points'],horizontal=True,color='#a3223a')
            st.table(factors[['feature','value','percentage points']],border='horizontal')
            st.caption(f"Probability scale · background {explanation['base_value']:.1%} · all 13 fields included · reconstruction error {explanation['reconstruction_error']:.2g}")
        else:
            st.info('Explanation unavailable: '+explanation['method'])
        st.download_button('Download assessed PDF',snapshot['pdf'],file_name=safe_filename(snapshot['inputs']['patient_id']),mime='application/pdf',on_click='ignore')

elif page=='Model evidence':
    st.subheader('How was this model evaluated?')
    st.write(info['evaluation_scope']); st.write(info['selection_rule'])
    st.caption('Complete procedures include preprocessing and calibration. Nested outer folds estimate selection performance; candidate OOF scores are selection evidence, not unbiased final estimates. Fold SD is dispersion, not a confidence interval.')
    m=info['metrics']; cols=st.columns(3)
    cols[0].metric('Sensitivity',f"{m['tp']}/{m['tp']+m['fn']} ({m['sensitivity']:.0%})")
    cols[1].metric('Specificity',f"{m['tn']}/{m['tn']+m['fp']} ({m['specificity']:.0%})")
    cols[2].metric('Brier score',f"{m['brier']:.3f}")
    st.write(f"Missed positives: {m['fn']} of {m['tp']+m['fn']}. False positives: {m['fp']} of {m['tn']+m['fp']}.")
    st.dataframe(pd.DataFrame([[m['tn'],m['fp']],[m['fn'],m['tp']]],index=['Actual absent','Actual present'],columns=['Predicted absent','Predicted present']))
    st.caption(f"Final ROC-AUC {m['roc_auc']:.3f} · Average precision {m['average_precision']:.3f} · Precision {m['precision']:.1%}")
    st.json(info['uncertainty'],expanded=False)
    with st.expander('Final ROC, precision-recall and calibration figures'):
        for title,filename in info['figures'].items():
            try:
                st.image(client.figure(filename,info['artifact_id']),caption=title,width='stretch')
            except Exception:
                st.info(title+' unavailable for this artifact.')
    try:
        ev=client.evaluation()
        if ev['artifact_id']!=info['artifact_id']:
            raise ValueError('Artifact changed')
        st.subheader('Development candidate comparison')
        st.dataframe(ev['candidates'],hide_index=True)
        st.subheader('Final reliability bins')
        st.dataframe(ev['reliability'],hide_index=True)
        with st.expander('Nested development folds and descriptive subgroups'):
            st.json({'nested_cv':ev['nested_cv'],'subgroups':ev['subgroups']})
        st.caption('Small subgroup denominators are suppressed. These results do not establish fairness or generalization.')
    except Exception:
        st.info('Evaluation tables unavailable from the serving source.')
    with st.expander('Data and model card'):
        st.json({'data_quality':info['data_quality'],'training_support':info['training_support'],'configuration':info['config']})
        st.write('Source: UCI Heart Disease, DOI 10.24432/C52P4X, CC BY 4.0. Source reconciliation verifies every available field. Specialist measurements are required. No external cohort validation has been completed.')

else:
    st.subheader('Research playground')
    st.warning('Associations only: changing an input is not evidence that an intervention changes disease status. The deployed threshold stays at 0.50.')
    try:
        ev=client.evaluation()
    except Exception:
        st.error('Research tables are unavailable from the serving source. Retry when the service is available.')
        st.stop()
    if ev['artifact_id']!=info['artifact_id']:
        st.error('Artifact changed. Refresh before exploring.'); st.stop()
    threshold=st.slider('Experimental threshold (development OOF predictions)',0.,1.,.5,.01)
    table=pd.DataFrame(ev['thresholds']); row=table.iloc[(table.threshold-threshold).abs().argmin()]
    st.write(f"Sensitivity {row.sensitivity:.0%}; specificity {row.specificity:.0%}; FP {int(row.fp)}; FN {int(row.fn)}")
    st.line_chart(table.set_index('threshold')[['sensitivity','specificity']])
    st.subheader('One-field synthetic scenario comparison')
    field=st.selectbox('Change one field',ALL_FEATURES,format_func=lambda x:FEATURE_LABELS[x])
    if field in CATEGORY_OPTIONS:
        value=st.selectbox('Scenario value',list(CATEGORY_OPTIONS[field]),format_func=lambda x:CATEGORY_OPTIONS[field][x])
    else:
        lo,hi,step=NUMERIC_RANGES[field]; cast=int if field in ['age','ca'] else float
        value=st.number_input('Scenario value '+UNITS.get(field,''),min_value=cast(lo),max_value=cast(hi),value=cast(DEMO[field]),step=cast(step))
    if st.button('Compare synthetic scenarios'):
        try:
            a=client.predict(DEMO,False); b=client.predict({**DEMO,field:value},False)
            if a['artifact_id']!=b['artifact_id'] or a['artifact_id']!=info['artifact_id']:
                raise ValueError('Artifact changed')
            st.write(f"Changed {FEATURE_LABELS[field]}: {DEMO[field]} → {value}")
            st.dataframe([{'Scenario':name,'Probability':r['probability'],'Domain':r['domain_status'],'Warnings':'; '.join(r['warnings']),'Artifact':r['artifact_id']} for name,r in [('A',a),('B',b)]],hide_index=True)
            support=info['training_support'].get(field)
            values=list(CATEGORY_OPTIONS[field]) if field in CATEGORY_OPTIONS else sorted(set(int(v) if field in ['age','ca'] else float(v) for v in __import__('numpy').linspace(*support,12)))
            curve=[]
            for v in values:
                r=client.predict({**DEMO,field:v},False)
                if r['artifact_id']!=a['artifact_id']:
                    raise ValueError('Artifact changed')
                curve.append({'Value':CATEGORY_OPTIONS[field][v] if field in CATEGORY_OPTIONS else v,'Probability':r['probability']})
            st.dataframe(curve,hide_index=True)
            if field not in CATEGORY_OPTIONS:
                st.line_chart(pd.DataFrame(curve).set_index('Value'))
            st.caption(f"Artifact {a['artifact_id']} · calibration {a['calibration']} · fixed decision threshold {a['threshold']}")
        except Exception:
            st.error('Comparison unavailable; check inputs and serving source.')

with st.expander('Privacy and limitations'):
    st.write('Inputs, assessment and PDF remain in this browser session’s server memory. API mode sends entered fields and the optional report ID to the configured API. Clear session removes interface state; downloads remain on your device. No patient database or global prediction cache is used. Request logs contain IDs, status, timing and artifact identity, not clinical fields. Use synthetic records on public demos. No legal-compliance claim is made.')
