"""Actual service and source authority — composed mathematical operators."""
import math
from pathlib import Path
import numpy as np
import pytest
from atlas.api import AtlasService
from atlas.discovery import discover, predict
from atlas.importers import decode_document
from atlas.memory import EvidenceStore
from atlas.graph import graph

ROOT=Path(__file__).resolve().parent.parent


def populated(tmp_path):
    db=EvidenceStore(tmp_path/'evidence.sqlite3')
    ids=[]
    for name,xy in [('calibration.csv',('coordinate','reading')),
                    ('secondary_calibration.csv',('measurement','calibrated'))]:
        text,sha,mime=decode_document(name,(ROOT/'examples'/'fixtures'/name).read_bytes())
        src=db.ingest(name,text,media_type=mime,original_sha256=sha)
        model=db.learn_equation(src['source_id'],*xy)
        ids.append(model['equation_id'])
    return db,ids


def test_composition_cross_source_replay_and_restart(tmp_path):
    db,ids=populated(tmp_path)
    first=db.compose_equations(ids,.375)
    assert first['supported'] and len(first['steps'])==2
    x=1.4-.5*.375**2+2*math.sin(.75)
    truth=.9+1.25*x+.08*x*x
    assert abs(first['value']-truth)<.02
    assert len(set(first['source_ids']))==2
    assert all(len(step['source_sha256'])==64 for step in first['steps'])
    restarted=EvidenceStore(tmp_path/'evidence.sqlite3')
    assert restarted.compose_equations(ids,.375)==first


def test_domain_checks_each_stage(tmp_path):
    db,ids=populated(tmp_path)
    first=db.compose_equations(ids,10)
    assert not first['supported'] and len(first['steps'])==1
    second=db.compose_equations(ids,2)
    assert not second['supported'] and len(second['steps'])==2
    assert second['steps'][0]['supported'] and not second['steps'][1]['supported']
    assert 'Outside' in second['reason']


def test_invalid_composition_does_not_mutate_memory(tmp_path):
    db,ids=populated(tmp_path)
    initial=db.stats()
    for bad in ([ids[0]], ids*4, [ids[0]]*2, [1.5,ids[1]], [True,ids[1]], [0,ids[1]]):
        with pytest.raises(ValueError):db.compose_equations(bad,.3)
    for x in (float('nan'),float('inf'),'nan'):
        with pytest.raises(ValueError):db.compose_equations(ids,x)
    assert initial==db.stats()


def test_service_router_with_exact_trace_and_graph(tmp_path):
    import shutil
    shutil.copytree(ROOT/'models'/'starter',tmp_path/'models'/'starter')
    srv=AtlasService(tmp_path)
    models=[]
    for name,cols in [('calibration.csv',('coordinate','reading')),
                      ('secondary_calibration.csv',('measurement','calibrated'))]:
        txt,sha,mime=decode_document(name,(ROOT/'examples'/'fixtures'/name).read_bytes())
        import base64
        doc=srv.process('ingest_file',{'filename':name,'base64':base64.b64encode((ROOT/'examples'/'fixtures'/name).read_bytes()).decode()})
        models.append(srv.process('learn_equation',{'source_id':doc['source_id'],'x_column':cols[0],'y_column':cols[1]})['equation_id'])
    answer=srv.process('respond',{'question':f'chain equations {models[0]} {models[1]} at 0.375'})
    assert answer['route']=='composed_equations' and answer['supported']
    assert 'composition' in answer['trace']
    assert any(n['id']=='composition' for n in graph()['nodes'])
    events=srv.events.tail(10)
    assert all('question' not in e and 'document' not in e for e in events)


def test_compositional_basis_recovers_interaction_held_out():
    rng=np.random.default_rng(81)
    x=rng.uniform(-2,2,120)
    y=.8+1.4*x*np.sin(2*x)+.45*np.cos(x)+rng.normal(0,.007,120)
    fit=discover(list(zip(x,y)))
    assert 'x·sin(2x)' in fit['terms']
    test=rng.uniform(-2,2,90)
    truth=.8+1.4*test*np.sin(2*test)+.45*np.cos(test)
    error=float(np.sqrt(np.mean((np.asarray(predict(fit,test))-truth)**2)))
    assert error<.03
    assert fit['blocked_holdout_rmse']<.1
    assert fit['stored_scalars']<=3


def test_unsupported_noise_has_no_exact_math_claim():
    rng=np.random.default_rng(3)
    x=rng.uniform(-2,2,120);y=rng.normal(0,1,120)
    fit=discover(list(zip(x,y)))
    assert fit['validation_rmse']>0.5
    assert 'physical law' in fit['limitation']


def test_extreme_coordinates_refused_for_numerical_stability():
    from atlas.discovery import discover
    with pytest.raises(ValueError,match='rescale'):
        discover([(1e20+i, float(i)) for i in range(20)])
