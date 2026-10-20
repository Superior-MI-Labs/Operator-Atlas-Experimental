"""Release v0.9: ordinary evidence usage, numerical stability, strict refusal."""
import json
from pathlib import Path
import numpy as np
import pytest
from atlas.api import AtlasService
from atlas.discovery import discover,predict
from atlas.examples import evidence_quality_study
from atlas.evaluation import coordinate_family_study

ROOT=Path(__file__).resolve().parents[1]

def test_public_qa_before_after_and_conflicts(tmp_path):
    r=evidence_quality_study(ROOT,tmp_path/'qa')
    assert r['positive_correct']==r['positive_total']==18
    assert r['negative_abstained']==r['negative_total']==10
    assert r['conflict_reported'] and r['before_all_abstained']

def test_unknown_attribute_refused_even_if_entity_matches(tmp_path):
    service=AtlasService(tmp_path)
    service.db.ingest('Source A','The reference prism was stored at KAPPA-47 for routine calibration.')
    assert not service.process('respond',{'question':'What color was the reference prism?'})['supported']

def test_old_discovery_model_serialization_unchanged():
    # Existing fitted models have no coordinate metadata. The prediction path
    # must still treat their legacy equation features as unnormalized.
    x=np.linspace(-2,2,50);y=.7+1.5*x
    fit=discover(list(zip(x,y)),basis_count=13)
    assert np.max(np.abs(np.asarray(predict(fit,x))-y)) < 1e-4
    old={k:v for k,v in fit.items() if k not in ('coordinate_center','coordinate_scale')}
    assert predict(old,[.2,.7])==pytest.approx(predict(fit,[.2,.7]))

def test_logarithmic_and_far_shifted_coordinate_study():
    r=coordinate_family_study()
    assert r['mean_unseen_rmse_16'] < r['mean_unseen_rmse_13']*.10
    assert r['large_coordinate_max_abs_error'] < 1e-6

def test_nonpositive_data_cannot_select_positive_only_terms():
    x=np.linspace(-2,2,32);y=x*x
    with pytest.raises(ValueError,match='Positive-only'):
        discover(list(zip(x,y)),basis_count=16)

def test_legacy_equation_in_database_still_predicts(tmp_path):
    import sqlite3
    s=AtlasService(tmp_path)
    a=s.db.ingest('legacy','This source stores the legacy calculation used in regression testing.')
    x=np.linspace(-2,2,40);y=.3+2*x
    fit=discover(list(zip(x,y)),basis_count=13)
    old={k:v for k,v in fit.items() if not k.startswith('coordinate_')}
    old.update(x_min=-2,x_max=2)
    with s.db.connect() as con:
        con.execute('INSERT INTO equations(source_id,x_column,y_column,method_version,model_json,created_at) VALUES(?,?,?,?,?,?)',
                    (a['source_id'],'x','y','compositional-bic-v2',json.dumps(old),0))
        eid=con.execute('SELECT id FROM equations').fetchone()[0]
    r=s.db.predict_equation(eid,.5)
    assert r['supported'] and abs(r['value']-1.3)<1e-4
