#!/usr/bin/env python3
"""Check the delivered report and rasterize its final PDF; run only on PSR."""
from pathlib import Path
import ast, hashlib, json, re, subprocess, zipfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';LAYOUT=DATA/'layout';LAYOUT.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for source in (ROOT/'tools').glob('*.py'):
    ast.parse(source.read_text(),filename=str(source))
md=(ROOT/'REPORT_SLIDES_CN.md').read_text()
image_links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md)
assert len(image_links)==17
assert all((ROOT/link).is_file() for link in image_links)
manifest=json.loads((DATA/'figure_manifest.json').read_text())
assert len(manifest)==18
assert all((ROOT/'figures'/(row['figure']+'.png')).is_file() for row in manifest)
html=(ROOT/'REPORT_SLIDES_CN.html').read_text()
assert not re.search(r'class=["\'][^"\']*katex-error',html)
subprocess.run(['pdftotext','-bbox-layout',str(ROOT/'REPORT_SLIDES_CN.pdf'),str(LAYOUT/'text.xml')],check=True)
doc=ET.parse(LAYOUT/'text.xml')
ns={'x':'http://www.w3.org/1999/xhtml'}
pages=doc.findall('.//x:page',ns);outside=[]
for i,page in enumerate(pages,1):
    width=float(page.attrib['width']);height=float(page.attrib['height'])
    for word in page.findall('.//x:word',ns):
        box=[float(word.attrib[key]) for key in ['xMin','yMin','xMax','yMax']]
        if box[0]<-1 or box[1]<-1 or box[2]>width+1 or box[3]>height+1:
            outside.append({'page':i,'text':word.text,'bbox':box})
assert len(pages)==36 and not outside,(len(pages),outside)
with zipfile.ZipFile(ROOT/'REPORT_SLIDES_CN.pptx') as pptx:
    slide_names=[name for name in pptx.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',name)]
    assert len(slide_names)==36
with (LAYOUT/'raster.log').open('w') as log:
    subprocess.run(['pdftoppm','-scale-to','800','-png','-r','70',
                    str(ROOT/'REPORT_SLIDES_CN.pdf'),str(LAYOUT/'page')],stdout=log,stderr=log,check=True)
raster=sorted(LAYOUT.glob('page-*.png'))
assert len(raster)==36
for sheet in range(3):
    canvas=Image.new('RGB',(1600,3120),'#d5e0e8');draw=ImageDraw.Draw(canvas)
    for k,path in enumerate(raster[sheet*12:(sheet+1)*12]):
        with Image.open(path) as img:
            panel=ImageOps.contain(img.convert('RGB'),(784,478))
        x=(k%2)*800+8;y=(k//2)*520+28
        canvas.paste(panel,(x,y))
        draw.text((x,y-20),'Slide '+str(sheet*12+k+1),fill='#17324d')
    canvas.save(LAYOUT/('contact-%02d.png'%(sheet+1)))
layout=dict(pages=len(pages),outside=outside,pdf_sha256=sha(ROOT/'REPORT_SLIDES_CN.pdf'))
(LAYOUT/'audit.json').write_text(json.dumps(layout,indent=2)+'\n')
metrics=json.loads((DATA/'report_metrics.json').read_text())
provenance=json.loads((DATA/'source_provenance.json').read_text())
assert metrics['passed'] and provenance['passed']
assert all(metrics['external']['checks'].values())
result=dict(passed=True,pdf_pages=36,pptx_slides=36,diagnostic_figures=18,
            markdown_image_links_checked=len(image_links),pdf_text_outside_page=outside,
            public_optics_checks=metrics['external']['checks'],
            source_provenance_passed=provenance['passed'],
            python_sources_parse=True,
            artifacts={name:sha(ROOT/name) for name in ['REPORT_SLIDES_CN.md','REPORT_SLIDES_CN.html',
                                                       'REPORT_SLIDES_CN.pdf','REPORT_SLIDES_CN.pptx']},
            scope='Report integrity and page-boundary checks on PSR; visual review remains separate.')
(DATA/'report_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
