#!/usr/bin/env python3
"""Final report, language, provenance and layout checks; run on PSR."""
from pathlib import Path
import ast,csv,json,hashlib,re,subprocess,zipfile,platform
import xml.etree.ElementTree as ET
import scipy,numpy,matplotlib
from PIL import Image,ImageDraw,ImageOps
R=Path(__file__).resolve().parents[1];D=R/'data';L=D/'layout';L.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in (R/'tools').glob('*.py'):ast.parse(p.read_text())
figs=sorted((R/'figures').glob('*.pdf'));assert len(figs)==8
texts={}
for p in figs:
 text=subprocess.check_output(['pdftotext',str(p),'-'],text=True)
 assert not re.search('[\u3400-\u9fff]',text),(p,'non-English label')
 texts[p.stem]=text
(D/'figure_text_english.json').write_text(json.dumps(texts,indent=2)+'\n')
subprocess.run(['pdftotext','-bbox-layout',str(R/'FIGURES_CN.pdf'),str(L/'text.xml')],check=True)
ns={'x':'http://www.w3.org/1999/xhtml'}
pages=ET.parse(L/'text.xml').findall('.//x:page',ns);outside=[]
for i,page in enumerate(pages,1):
 w,h=float(page.attrib['width']),float(page.attrib['height'])
 for word in page.findall('.//x:word',ns):
  b=[float(word.attrib[k]) for k in ['xMin','yMin','xMax','yMax']]
  if b[0]<-1 or b[1]<-1 or b[2]>w+1 or b[3]>h+1:outside.append([i,word.text])
assert len(pages)==8 and not outside
with zipfile.ZipFile(R/'FIGURES_CN.pptx') as z:
 slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
 assert len(slides)==8
with (L/'raster.log').open('w') as out:
 subprocess.run(['pdftoppm','-scale-to','1100','-png',str(R/'FIGURES_CN.pdf'),str(L/'page')],stdout=out,stderr=out,check=True)
canvas=Image.new('RGB',(1500,2040),'#dce5ee');draw=ImageDraw.Draw(canvas)
for i,p in enumerate(sorted(L.glob('page-*.png'))):
 with Image.open(p) as img:panel=ImageOps.contain(img.convert('RGB'),(735,465))
 x=i%2*750+7;y=i//2*510+25;canvas.paste(panel,(x,y));draw.text((x,y-18),'Slide '+str(i+1),fill='black')
canvas.save(L/'contact.png')
terrain=list(csv.DictReader((D/'terrain_path_checks.csv').open()))
assert len(terrain)==4 and all(x['blocked']=='0' and x['inside_selected_face']=='1' for x in terrain)
inputs=json.loads((D/'input_provenance.json').read_text())
assert all(digest(Path(name))==value for name,value in inputs['input_sha256'].items())
old=R.parent/'interface-radio-report-20260912'
provenance=json.loads((old/'data/source_provenance.json').read_text())
rp=old/'external/RadioPropa-544a2d6c4e284e3d724cb741dc481245a0f633d7/radiopropa'
assert digest(old/'external/build-radiopropa/libradiopropa.so')==provenance['radiopropa']['library_sha256']
metrics=json.loads((D/'metrics.json').read_text());assert metrics['passed']
sp=json.loads((D/'sp_metrics.json').read_text());assert sp['passed']
assert all(digest(Path(name))==value for name,value in sp['source_sha256'].items())
audit=dict(passed=True,host=platform.node(),slides=8,figures=8,all_figure_labels_english=True,sp_checks_passed=True,
 page_text_outside=outside,terrain_checks=terrain,input_files_unchanged=True,
 versions=dict(scipy=scipy.__version__,numpy=numpy.__version__,matplotlib=matplotlib.__version__),
 public_library=provenance['radiopropa'],
 reference_sources={str(p):digest(p) for p in [rp/'src/module/PropagationCK.cpp',rp/'src/ParticleState.cpp',rp/'src/module/Discontinuity.cpp']},
 artifacts={p.name:digest(p) for p in R.glob('FIGURES_CN.*')},
 scope='No local tests or production edits. Saved real-shower replay and public-library checks ran on PSR.')
(D/'report_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['passed','slides','figures','all_figure_labels_english','page_text_outside','versions']},indent=2))
