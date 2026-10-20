"""Documentation -> train equation -> provenance -> prediction -> refusal tests."""
import csv, io, json, math, tempfile, zipfile
from pathlib import Path
import pytest
from atlas.api import AtlasService
from atlas.doctor import run_doctor
from atlas.examples import fixture_demonstration
from atlas.memory import EvidenceStore
from atlas.importers import decode_document

ROOT=Path(__file__).resolve().parent.parent

def test_fixture_reproducibility_and_negative_controls():
    with tempfile.TemporaryDirectory() as d:
        result=fixture_demonstration(ROOT,Path(d))
        assert not result['before']['found'] and result['after']['found']
        assert result['after']['source_ids']==[1]
        assert result['equation']['absolute_error']<.005
        assert result['equation']['random_control_refused']
        assert result['equation']['outside_range_refused']

def _store(tmp_path):
    m=EvidenceStore(tmp_path/'data.sqlite3')
    b=(ROOT/'examples'/'fixtures'/'calibration.csv').read_bytes()
    text,digest,mime=decode_document('calibration.csv',b)
    return m,m.ingest('calibration.csv',text,media_type=mime,original_sha256=digest)

def test_equation_persists_restart_and_replay(tmp_path):
    m,source=_store(tmp_path)
    eq=m.learn_equation(source['source_id'],'coordinate','reading')
    assert not eq['reused']
    after=EvidenceStore(tmp_path/'data.sqlite3')
    eq2=after.learn_equation(source['source_id'],'coordinate','reading')
    assert eq2['reused'] and eq2['equation_id']==eq['equation_id']
    val=after.predict_equation(eq['equation_id'],.375)
    assert val['supported'] and abs(val['value']-(1.4-.5*.375**2+2*math.sin(2*.375)))<.005
    assert val['source_sha256']==source['sha256']
    assert after.predict_equation(eq['equation_id'],-10)['value'] is None

def test_bad_csv_rejected_and_preserves_db(tmp_path):
    m,source=_store(tmp_path)
    for xcol,ycol in [('coordinate','coordinate'),('missing','reading'),('coordinate','nonexistent')]:
        with pytest.raises(ValueError):m.learn_equation(source['source_id'],xcol,ycol)
    malformed='coordinate,reading\n'+'\n'.join(f'{i},{i}' for i in range(10))+'\n'
    bad=m.ingest('short.csv',malformed,media_type='text/csv')
    with pytest.raises(ValueError,match='12'):m.learn_equation(bad['source_id'],'coordinate','reading')
    bad2=m.ingest('invalid.csv','coordinate,reading\n'+ '\n'.join(f'{i},NaN' for i in range(20)),media_type='text/csv')
    with pytest.raises(ValueError,match='finite'):m.learn_equation(bad2['source_id'],'coordinate','reading')
    assert m.list_equations()==[]

def test_random_control_and_lexical_scope(tmp_path):
    m,source=_store(tmp_path)
    b=(ROOT/'examples'/'fixtures'/'random_control.csv').read_bytes()
    text,digest,mime=decode_document('random_control.csv',b)
    bad=m.ingest('random_control.csv',text,media_type=mime)
    with pytest.raises(ValueError,match='No useful equation'):m.learn_equation(bad['source_id'],'coordinate','reading')
    assert not m.search('what is the answer to this question')

def test_service_actions_share_existing_authority(tmp_path):
    import shutil
    shutil.copytree(ROOT/'models'/'starter',tmp_path/'models'/'starter')
    s=AtlasService(tmp_path)
    b=(ROOT/'examples'/'fixtures'/'calibration.csv').read_bytes()
    import base64
    uploaded=s.process('ingest_file',{'filename':'calibration.csv','base64':base64.b64encode(b).decode()})
    fitted=s.process('learn_equation',{'source_id':uploaded['source_id'],'x_column':'coordinate','y_column':'reading'})
    listing=s.process('equations',{})
    predicted=s.process('predict_equation',{'equation_id':fitted['equation_id'],'x':.375})
    assert listing['equations'][0]['equation_id']==fitted['equation_id']
    assert predicted['supported']

def test_doctor_bundle_embeds_only_public_fixtures(tmp_path):
    out=tmp_path/'diag.zip'
    result=run_doctor(ROOT,out,quick=True)
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        fixtures=json.loads(z.read('fixture_demo.json'))
        assert fixtures['schema']=='operator-atlas-fixture-study-v1'
        assert fixtures['equation']['outside_range_refused']
        for filename in z.namelist():
            assert not filename.endswith('.sqlite3')
    assert result['result'] in ('PASS','FAIL')
