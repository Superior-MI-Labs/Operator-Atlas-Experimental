"""Developer support report must be actionable and must not leak private evidence."""
import json
import tempfile
import zipfile
from pathlib import Path
import pytest
from atlas.doctor import _sandbox, _integrity, run_doctor
from atlas.memory import EvidenceStore
ROOT=Path(__file__).resolve().parent.parent


def test_disposable_copy_excludes_user_content_and_logs():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        (base/'source').mkdir()
        # Seed a minimal distributable subset and a private document.
        for name in ('atlas','models/starter','models/dense-control','models/stacked-factorized',
                     'models/stacked-dense','tests','web','examples','tools','docs'):
            import shutil
            dst=base/'source'/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(ROOT/name,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('data/corpus.txt','pytest.ini','launch.sh','.gitignore','README.md','release.json','requirements.txt','requirements-train.txt','graph-desktop.png','graph-mobile.png'):
            dest=base/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        EvidenceStore(base/'source'/'data'/'evidence.sqlite3').ingest('Secret fixture',
                    'A_Private_Diagnostic_Test_String_That_Must_Never_Leave_The_Source')
        (base/'source'/'logs').mkdir()
        (base/'source'/'logs'/'events.jsonl').write_text('another_private_log_marker')
        isolated=base/'isolated';isolated.mkdir()
        _sandbox(base/'source',isolated)
        assert not (isolated/'data'/'evidence.sqlite3').exists()
        assert not (isolated/'logs').exists()
        assert not list(isolated.rglob('*.sqlite3'))


def test_release_integrity_detects_corrupted_file():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        (root/'atlas').mkdir()
        (root/'atlas'/'model.py').write_text('changed')
        (root/'SHA256SUMS.json').write_text(json.dumps({'algorithm':'sha256','files':{'atlas/model.py':'0'*64}}))
        with pytest.raises(AssertionError,match='integrity mismatch'):
            _integrity(root)


def test_quick_doctor_produces_reviewable_private_safe_zip():
    with tempfile.TemporaryDirectory() as td:
        dest=Path(td)/'support.zip'
        result=run_doctor(ROOT,dest,quick=True)
        assert dest.exists()
        # Normal CI validates the checksum; a developer tree may have edits,
        # so this test cares about a complete report, not claimed release health.
        assert result['result'] in ('PASS','FAIL')
        with zipfile.ZipFile(dest) as z:
            assert {'doctor.json','report.json','events.json','experiment_index.json','fixture_demo.json','equation_study.json','evidence_quality_study.json','coordinate_study.json','historical_sunspots.json'}==set(z.namelist())
            assert all(not n.endswith('.sqlite3') for n in z.namelist())
            doctor=json.loads(z.read('doctor.json'))
            assert doctor['mode']=='quick' and doctor['checks']
