"""Regression checks for honest, integrated response pathway."""
from pathlib import Path
import json
import pytest
from atlas.api import AtlasService

ROOT=Path(__file__).resolve().parents[1]
QUESTION='Where was the Cloudglass Observatory reference prism stored?'

def service(tmp_path):
    # No changes to user's evidence DB or user checkpoint.
    return AtlasService(tmp_path)

def test_before_after_fictional_source(tmp_path):
    s=service(tmp_path)
    before=s.process('respond',{'question':QUESTION})
    assert before['route']=='abstain' and not before['supported']
    assert 'KAPPA' not in before['response']
    # Mirror real file import from shipped fixture through one existing DB.
    text=(ROOT/'examples/fixtures/research_notebook.md').read_text()
    src=s.process('ingest',{'title':'Fictional notebook','text':text})
    after=s.process('respond',{'question':QUESTION})
    assert after['route']=='source_excerpt' and after['supported']
    assert 'KAPPA-47' in after['response']
    assert after['evidence'][0]['source_id']==src['source_id']
    assert 'not a verified answer' in after['disclosure']
    assert not s.process('respond',{'question':'What is the unrelated ryztodon nebula size?'})['supported']

def test_equation_route_and_out_of_domain(tmp_path):
    s=service(tmp_path)
    import csv
    from atlas.importers import decode_document
    f=ROOT/'examples/fixtures/calibration.csv'
    text,sha,mime=decode_document(f.name,f.read_bytes())
    source=s.db.ingest('calibration.csv',text,media_type=mime)
    eq=s.process('learn_equation',{'source_id':source['source_id'],'x_column':'coordinate','y_column':'reading'})
    r=s.process('respond',{'question':f"Predict equation {eq['equation_id']} at 0.375"})
    assert r['route']=='qualified_equation' and r['supported']
    assert r['equation']['source_id']==source['source_id']
    bad=s.process('respond',{'question':f"Evaluate equation #{eq['equation_id']} at x=123456"})
    assert bad['route']=='qualified_equation' and not bad['supported']
    assert 'cannot evaluate' in bad['response'].lower()

def test_explicit_neural_mode_requires_model(tmp_path):
    s=service(tmp_path)
    with pytest.raises(ValueError,match='unavailable'):
        s.process('respond',{'question':'Hello','mode':'experimental'})
    with pytest.raises(ValueError):s.process('respond',{'question':'hello','mode':'auto'})
    with pytest.raises(ValueError):s.process('respond',{'question':' '})
    with pytest.raises(ValueError):s.process('load_example',{'name':'../../secret.txt'})

def test_fixture_loader_is_idempotent_and_private_paths_rejected(tmp_path):
    import shutil
    shutil.copytree(ROOT/'examples',tmp_path/'examples')
    s=service(tmp_path)
    one=s.process('load_example',{'name':'research_notebook.md'})
    two=s.process('load_example',{'name':'research_notebook.md'})
    assert one['source_id']==two['source_id']
    assert s.db.stats()['source_count']==1
    assert s.process('respond',{'question':QUESTION})['route']=='source_excerpt'


def test_multiple_sources_are_not_silently_reconciled(tmp_path):
    s=service(tmp_path)
    s.process('ingest',{'title':'Fictional source A','text':'Cloudglass Observatory reference prism stored in cabinet KAPPA-47 during the inspection.'})
    s.process('ingest',{'title':'Fictional source B','text':'Cloudglass Observatory reference prism stored in cabinet OMEGA-52 during the inspection.'})
    r=s.process('respond',{'question':QUESTION})
    assert r['route']=='source_excerpt'
    assert len({x['source_id'] for x in r['evidence']})==2
    assert 'NOT been reconciled' in r['response']
    assert 'NOT reconciled' in r['disclosure']
