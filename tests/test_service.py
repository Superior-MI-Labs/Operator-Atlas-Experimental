import tempfile
from pathlib import Path
import pytest
from atlas.api import AtlasService
ROOT=Path(__file__).resolve().parent.parent

def test_service_full_workflow():
    s=AtlasService(ROOT); assert s.status()['model_loaded']
    res=s.process('generate',{'prompt':'The mathematical atlas','max_new':15})
    assert len(res['response'])==15
    fit=s.process('fit_equation',{'points':[[x,1+x] for x in range(14)]})
    assert fit['validation_rmse']<1e-4
    assert s.process('experiments',{'seed':11})['schema']==1
    with pytest.raises(ValueError):s.process('missing',{})

def test_reject_invalid_source():
    s=AtlasService(ROOT)
    with pytest.raises(ValueError):s.process('ingest',{'title':'X','text':'short'})
