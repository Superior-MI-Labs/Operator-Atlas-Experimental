#!/usr/bin/env python3
"""Mechanical audit: relative Markdown assets, release data pointers, public safety.

This audit does not establish that an experimental scientific claim is valid.
"""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parent.parent

def audit():
    errors=[];checked=0
    pages=[ROOT/'README.md',*(ROOT/'docs').glob('*.md'),ROOT/'AGENTS.md',ROOT/'CONTRIBUTING.md',ROOT/'SECURITY.md']
    for page in pages:
        body=page.read_text(encoding='utf-8')
        for raw in re.findall(r'(?<!!)\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)|<img\s+src="([^"]+)"',body):
            for link in raw:
                if not link or link.startswith(('http://','https://','#','mailto:','data:')):continue
                link=link.split('#')[0]
                if ' ' in link or '[' in link or ']' in link:continue
                checked+=1
                target=(page.parent/link).resolve()
                if not target.is_relative_to(ROOT.resolve()) or not target.exists():errors.append(f'Broken relative reference: {page.relative_to(ROOT)} -> {link}')
    # Numeric claims must come from published raw records; no per-run cherry picking.
    ab=json.loads((ROOT/'results/multi_seed_ablation.json').read_text())
    dense=[x['validation_loss'] for x in ab['runs'] if x['variant']=='dense_width38']
    fact=[x['validation_loss'] for x in ab['runs'] if x['variant']=='factorized_rank16']
    if len(dense)!=3 or len(fact)!=3 or abs(sum(dense)/3-0.58505)>.001 or abs(sum(fact)/3-1.01851)>.001:
        errors.append('README language statistics diverge from recorded multi-seed study')
    real=json.loads((ROOT/'results/historical_sunspots_v1.json').read_text())
    sr=str(real)
    if not all(v in sr for v in ['30.913','41.397','39.062']):errors.append('Historical benchmark numeric record changed: review claims')
    if errors:
        print('\n'.join(errors));return 1
    print(f'Public-doc audit PASS: {len(pages)} pages, {checked} local references, recorded metrics cross-checked')
    return 0

if __name__=='__main__':sys.exit(audit())
