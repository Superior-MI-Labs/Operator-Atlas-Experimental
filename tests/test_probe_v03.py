from pathlib import Path
import numpy as np
import pytest
from atlas.model import CharacterModel
from atlas.probe import probe_model
ROOT=Path(__file__).resolve().parent.parent

def test_score_consistent_with_current_execution_and_reject_short():
    m=CharacterModel(ROOT/'models/dense-control')
    original=m.score_text('The atlas learns new mathematical patterns.')
    from_copy=m.score_text('The atlas learns new mathematical patterns.',operators=dict(m.operators))
    assert original==from_copy
    with pytest.raises(ValueError):m.score_text('a')

def test_probe_validation_only_and_storage(tmp_path):
    import shutil
    (tmp_path/'models').mkdir()
    shutil.copytree(ROOT/'models/dense-control',tmp_path/'models/dense-control')
    (tmp_path/'data').mkdir()
    shutil.copy2(ROOT/'data/corpus.txt',tmp_path/'data/corpus.txt')
    result=probe_model(tmp_path,characters=120,allowed_increase=.05,write_candidate=True)
    assert result['selected_validation']['cross_entropy']<=result['base_validation']['cross_entropy']+.05+1e-8
    assert result['relative_payload_saving']>0
    assert result['partitions']['test_characters']==120
    assert result['partitions']['validation_characters']==120
    assert all('validation_loss' in a for r in result['operators'] for a in r['alternatives'])
    assert all('test_loss' not in a for r in result['operators'] for a in r['alternatives'])
    candidate=CharacterModel(result['candidate_path'])
    assert candidate.status()['parameter_bytes']==result['selected_payload_bytes']
    with pytest.raises(ValueError,match='already exists'):
        probe_model(tmp_path,characters=120,allowed_increase=.05,write_candidate=True)

def test_probe_reject_bad_budget():
    with pytest.raises(ValueError):probe_model(ROOT,allowed_increase=-.1)
    with pytest.raises(ValueError):probe_model(ROOT,characters=20)

def test_latency_aware_policy_returns_qualified_plan():
    result=probe_model(ROOT,characters=120,allowed_increase=.05,
                       objective='latency_guarded',max_slowdown=0)
    assert result['decision']['objective']=='latency_guarded'
    assert result['selected_validation']['cross_entropy']<=result['base_validation']['cross_entropy']+.05+1e-7
    if result['decision']['accepted']:
        assert result['timing']['candidate_to_baseline_ratio']<=1.0
    else:
        assert result['selected_payload_bytes']==result['original_payload_bytes']
        assert result['decision']['reason']=='reverted_to_dense_due_to_validation_latency'
        assert abs(result['selected_test']['cross_entropy']-result['base_test']['cross_entropy'])<1e-8


def test_probe_reject_unknown_objective_and_invalid_latency_budget():
    with pytest.raises(ValueError, match='objective'):
        probe_model(ROOT,objective='invented')
    with pytest.raises(ValueError, match='slowdown'):
        probe_model(ROOT,max_slowdown=-.01)
