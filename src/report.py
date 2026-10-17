"""Escaped, wrapped, in-memory PDF exports from an immutable assessment snapshot."""
import io
import re
from pathlib import Path
from xml.sax.saxutils import escape
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from .data_pipeline import ALL_FEATURES, CATEGORY_OPTIONS, FEATURE_LABELS
from .schemas import UNITS

FONT=Path(reportlab.__file__).parent/'fonts/Vera.ttf'
if 'HeartWise' not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(TTFont('HeartWise',str(FONT)))

def safe_text(value):
    # Represent glyphs absent from the embedded font explicitly rather than black squares.
    coverage=pdfmetrics.getFont('HeartWise').face.charToGlyph
    text=''.join(c if ord(c) in coverage or c in '\n\t' else f'[U+{ord(c):04X}]' for c in str(value))
    return escape(text)

def safe_filename(identifier):
    name=re.sub(r'[^A-Za-z0-9_-]+','_',identifier).strip('_')[:40] or 'anonymous'
    return 'heartwise_'+name+'.pdf'

def build_pdf_report(snapshot):
    patient,result=snapshot['inputs'],snapshot['result']
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=42)
    styles=getSampleStyleSheet()
    body=ParagraphStyle('HWBody',parent=styles['BodyText'],fontName='HeartWise',fontSize=9,leading=13,spaceAfter=6)
    title=ParagraphStyle('HWTitle',parent=body,fontSize=23,leading=28,textColor=colors.HexColor('#92233A'),spaceAfter=14)
    heading=ParagraphStyle('HWHeading',parent=body,fontSize=13,leading=17,spaceBefore=12,spaceAfter=8)
    def p(value,style=body):
        return Paragraph(safe_text(value),style)
    def table(rows,widths):
        t=Table([[p(c) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#F5E8EB')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#92233A'))]))
        return t
    story=[p('HEARTWISE',title),p('Research assessment report',heading),
           p('Educational demonstration of disease status in the Cleveland dataset. Not a diagnosis, future heart-attack forecast or treatment recommendation.'),
           p(f"Report ID: {patient.get('patient_id','Anonymous')}"),p(f"Assessed: {snapshot['assessed_at']} (UTC)"),
           p(f"Artifact: {result['artifact_id']}"),p('Model output',heading)]
    probability='Withheld' if result['probability'] is None else f"{result['probability']:.1%}"
    story.append(table([['Measure','Value'],['Disease-present score',probability],['Classification',result['prediction_label']],
                        ['Calibration',result['calibration']],['Decision threshold',str(result['threshold'])],['Domain status',result['domain_status']]], [170,341]))
    story.append(p('Positive class means UCI disease status num > 0. The fixed 0.50 threshold is a research policy, not a clinically validated decision rule. No clinical risk bands are assigned.'))
    for warning in result['warnings']:
        story.append(p('Domain warning: '+warning))
    story.append(p('Within-range values do not establish support for the complete profile. Individual probability uncertainty has not been quantified.'))
    story.append(p('Assessed inputs',heading))
    rows=[['Field','Value','Unit']]
    for field in ALL_FEATURES:
        value=CATEGORY_OPTIONS[field][patient[field]] if field in CATEGORY_OPTIONS else patient[field]
        rows.append([FEATURE_LABELS[field],value,UNITS.get(field,'Category')])
    story.append(table(rows,[205,220,86]))
    story.append(p('Explanation of this model score',heading)); explanation=result['explanation']
    story.append(p(explanation['note']))
    if explanation['status']=='available':
        story.append(p(f"Target: {explanation['target']}; scale: probability; background mean: {explanation['base_value']:.4f}."))
        story.append(p(explanation['method']))
        rows=[['Field (selected value)','Contribution (percentage points)']]
        rows += [[f"{f['feature']} ({f['value']})",f"{100*f['impact']:+.2f}"] for f in explanation['contributions']]
        story.append(table(rows,[370,141]))
        story.append(p('All raw-field contributions are included; inactive encoded categories are not discarded.'))
    else:
        story.append(p('Explanation unavailable: '+explanation['method']))
    story += [Spacer(1,10),p('Privacy: inputs and this PDF stay in session memory. In API mode, clinical fields and the optional report ID are sent to the configured API. Clear session removes the assessment from the interface; downloaded copies remain with you.'),
              p('Some characters outside the embedded font are represented by their Unicode code point. Internal retrospective evaluation is not external or clinical validation.')]
    def footer(canvas,doc):
        canvas.setFont('HeartWise',8)
        canvas.drawString(42,24,'HeartWise | Educational research only')
        canvas.drawRightString(A4[0]-42,24,f'Page {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()
