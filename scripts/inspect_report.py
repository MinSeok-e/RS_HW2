"""Render PDF pages for visual inspection and check essential text."""
import json
import os
from pathlib import Path
import pypdfium2
from pypdf import PdfReader
if not os.getenv('SLURM_JOB_ID'):
    raise RuntimeError('Use a compute node')
root = Path('/home/minseok/data/semester/1/RS/RS_HW2')
status = json.loads((root / 'report/status.json').read_text())
pdf = Path(status['pdf'])
reader = PdfReader(str(pdf))
text = '\n'.join(page.extract_text() for page in reader.pages)
for required in ('2026713004', 'Minseok Jeong', '0.11419', '0.09812', 'Q1.', 'Q2.'):
    assert required in text, f'Missing report text: {required}'
previews = root / 'report/previews'
previews.mkdir(exist_ok=True)
document = pypdfium2.PdfDocument(str(pdf))
for i, page in enumerate(document):
    page.render(scale=1.2).to_pil().save(previews / f'page-{i+1}.png')
(root / 'report/text-check.txt').write_text(text)
print(json.dumps({'pages': len(reader.pages), 'required_text': 'passed', 'pdf': str(pdf)}, indent=2))
