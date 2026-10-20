import json
from pathlib import Path
from atlas.evaluation import real_world_sunspot_study

ROOT=Path(__file__).resolve().parents[1]

def test_historical_study_stable_and_chronological():
    a=real_world_sunspot_study(ROOT); b=real_world_sunspot_study(ROOT)
    assert a==b and a['rows']==309 and len(a['runs'])==6
    assert all(r['future_count']==15 and r['train_count']>=180 for r in a['runs'])
    assert a['sha256']

def test_history_is_not_a_favorable_synthetic_fixture():
    a=real_world_sunspot_study(ROOT)
    assert any(r['equation_16_rmse']>r['historical_mean_rmse'] for r in a['runs'])
    assert 'Off-domain stress' in a['note']

def test_full_package_has_release_docs():
    assert (ROOT/'docs/WHITEPAPER_V1.md').is_file()
    assert (ROOT/'assets/architecture_v1.png').is_file()
    assert json.loads((ROOT/'release.json').read_text())['version']=='1.0.0'

def test_publish_script_is_checksum_whitelist_only():
    content=(ROOT/'tools/publish_github.sh').read_text()
    assert "manifest['files']" in content and 'SHA256SUMS.json' in content
    assert 'git push origin HEAD:main' in content and '--force' not in content
