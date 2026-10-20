"""One-command, local-only diagnostic qualification and support bundle.

Tests execute against a disposable copy. Private evidence, logs and candidate
artifacts never enter the copy or the exported diagnostic report.
"""
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .diagnostics import export_report

# Whitelist-only: only source code, shipped checkpoints, test fixtures, and UI.
_COPY_DIRS = ('atlas', 'models/starter', 'models/dense-control', 'models/stacked-factorized', 'models/stacked-dense', 'tests', 'web', 'examples', 'tools', 'docs', 'assets')
_COPY_FILES = ('data/corpus.txt', 'pytest.ini', 'launch.sh', '.gitignore', 'README.md', 'release.json', 'requirements.txt', 'requirements-train.txt', 'graph-desktop.png', 'graph-mobile.png')


def _integrity(root):
    """Check shipped files against the release checksum manifest, excluding user data."""
    manifest_file=root/'SHA256SUMS.json'
    if not manifest_file.is_file():
        raise FileNotFoundError('Release checksum manifest missing')
    doc=json.loads(manifest_file.read_text())
    if doc.get('algorithm')!='sha256' or not isinstance(doc.get('files'),dict):
        raise ValueError('Invalid checksum manifest')
    changed=[]
    for name, expected in doc['files'].items():
        path=root/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or
            not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected):
            changed.append(name)
    if changed:
        raise AssertionError('Release integrity mismatch: '+', '.join(changed[:8]))
    return {'checked_files':len(doc['files']),'release_version':doc.get('version')}


def _sandbox(root, temp):
    for name in _COPY_DIRS:
        src = root / name
        if not src.is_dir():
            if name == 'assets':
                continue  # Public derivative figures are not an execution dependency.
            raise FileNotFoundError('Required component missing: ' + name)
        dst = temp / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.pyc', '*.sqlite3*'))
    for name in _COPY_FILES:
        src = root / name
        if not src.is_file():
            raise FileNotFoundError('Required component missing: ' + name)
        dst = temp / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # Only shipped public JSON research results; never copy runtime run history.
    (temp/'results').mkdir(exist_ok=True)
    for src in (root/'results').glob('*.json'):
        shutil.copy2(src,temp/'results'/src.name)


def _run(name, fn, checks):
    started = time.perf_counter()
    try:
        values = fn() or {}
        checks.append({'name': name, 'status': 'PASS', 'seconds': round(time.perf_counter()-started, 3), 'measurements': values})
    except Exception as exc:
        # Tests run on packaged fixtures, never on user documents. Keep only
        # bounded alphanumeric failure codes, not arbitrary exception output.
        detail=re.sub(r'[^a-zA-Z0-9_./: ()=,\[\]-]', '', str(exc))[:240]
        detail=re.sub(r'/tmp/[^ /]+', '[temporary]', detail)
        checks.append({'name': name, 'status': 'FAIL', 'seconds': round(time.perf_counter()-started, 3),
                       'reason': type(exc).__name__, 'detail': detail})


def _numerical(temp):
    import numpy as np
    from .model import CharacterModel
    from .operators import Operator, factorize, quantize
    from .discovery import discover, predict
    rng = np.random.default_rng(1729)
    a = rng.standard_normal((32, 4)).astype('float32')
    b = rng.standard_normal((4, 32)).astype('float32')
    matrix = a @ b
    x = rng.standard_normal((64, 32)).astype('float32')
    low = factorize('test', matrix, 4)
    error = float(np.linalg.norm(low.apply(x) - x @ matrix.T) / np.linalg.norm(x @ matrix.T))
    assert error < 1e-5, 'Low-rank operator failed parity'
    q = quantize('test', matrix)
    qerror = float(np.linalg.norm(q.apply(x) - x @ matrix.T) / np.linalg.norm(x @ matrix.T))
    assert qerror < .03, 'Quantized operator failed expected accuracy'
    try:
        Operator('bad', 'factorized', (3, 4), {'a': np.zeros((3,2)), 'b': np.zeros((1,4))})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid factor contract was accepted')
    for name in ('starter', 'dense-control', 'stacked-factorized', 'stacked-dense'):
        m = CharacterModel(temp/'models'/name)
        r1 = m.generate('The atlas learns', 12, seed=123)
        r2 = m.generate('The atlas learns', 12, seed=123)
        assert r1 == r2 and len(r1['response']) == 12
        assert m.manifest['training']['numpy_torch_max_error'] < 1e-4
    xs = np.linspace(-2,2,70)
    ys = 1.3 - .4*xs**2 + 1.2*np.sin(2*xs)
    eq = discover(list(zip(xs,ys)))
    pred = np.asarray(predict(eq,[-.7,0,.8]))
    truth = 1.3-.4*np.array([-.7,0,.8])**2+1.2*np.sin(2*np.array([-.7,0,.8]))
    assert float(np.max(np.abs(pred-truth))) < .0001
    return {'rank4_relative_error': error, 'int8_relative_error': qerror,
            'formula_validation_rmse': eq['validation_rmse'], 'checkpoints_tested': 4}


def _equation_study(temp):
    from .evaluation import equation_family_study
    return equation_family_study()

def _language(temp):
    from .language_eval import evaluate_language
    return evaluate_language(temp)


def _workflows(temp):
    from .api import AtlasService
    from .graph import graph
    service = AtlasService(temp)
    assert service.db.stats()['source_count'] == 0, 'Isolated workspace was contaminated'
    assert service.status()['model_loaded']
    nodes = {item['id'] for item in graph()['nodes']}
    assert nodes and all(a['from'] in nodes and a['to'] in nodes for a in graph()['edges'])
    first = service.process('ingest', {'title':'Diagnostic fixture',
        'text':'Spectral harmonics are mathematically structured. This is disposable test evidence only.'})
    assert first['source_id']
    answer = service.process('ask_sources',{'question':'spectral harmonics'})
    assert answer['found'] and answer['passages']
    assert not service.process('ask_sources',{'question':'unmatchedxyztoken'})['found']
    generated = service.process('generate',{'prompt':'An atlas', 'max_new':10, 'seed':23})
    assert len(generated['response'])==10
    assert service.db.stats()['source_count']==1
    return {'graph_nodes':len(nodes),'source_fixture':'passed','generation_characters':10}


def _fixture_learning(temp):
    from .examples import fixture_demonstration
    result=fixture_demonstration(temp,temp/'fixture-study')
    return result


def _http(temp):
    # Short-lived loopback server, independent of any server the user is running.
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    env = {**os.environ, 'PYTHONPATH':str(temp), 'PYTHONUNBUFFERED':'1'}
    cmd = [sys.executable, '-m', 'atlas.cli', 'serve', '--host', '127.0.0.1', '--port', str(port)]
    proc = subprocess.Popen(cmd,cwd=temp,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    base=f'http://127.0.0.1:{port}'
    try:
        ready=False
        for _ in range(80):
            if proc.poll() is not None: break
            try:
                with urlopen(base+'/api/status',timeout=1) as res:
                    stat=json.load(res)
                    ready=stat['model_loaded']
                    break
            except OSError:
                time.sleep(.1)
        assert ready, 'Temporary loopback server did not start'
        with urlopen(base+'/api/graph',timeout=4) as res:
            assert len(json.load(res)['nodes'])>=11
        with urlopen(base+'/',timeout=4) as res:
            assert b'html' in res.read(1000).lower()
        payload=json.dumps({'prompt':'atlas','max_new':8,'seed':2}).encode()
        req=Request(base+'/api/generate',data=payload,headers={'Content-Type':'application/json'},method='POST')
        with urlopen(req,timeout=6) as res:
            assert json.load(res)['generated_characters']==8
        # A foreign website must not issue state-mutating actions.
        hostile=Request(base+'/api/generate',data=payload,headers={'Content-Type':'application/json','Origin':'https://evil.example'},method='POST')
        from urllib.error import HTTPError
        try:
            urlopen(hostile,timeout=4)
        except HTTPError as err:
            assert err.code==403
        else:
            raise AssertionError('Foreign Origin accepted')
        return {'loopback_routes_checked':4,'foreign_origin_rejected':True}
    finally:
        proc.terminate()
        try:proc.wait(timeout=4)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)


def _pytest(temp):
    if importlib.util.find_spec('pytest') is None:
        return {'status':'SKIP','reason':'pytest_not_installed','tests':None}
    env={**os.environ,'PYTHONPATH':str(temp),'PYTHONDONTWRITEBYTECODE':'1'}
    proc=subprocess.run([sys.executable,'-m','pytest','-q','--tb=line','-p','no:cacheprovider','tests'],
          cwd=temp,env=env,capture_output=True,text=True,timeout=100)
    tail=(proc.stdout+'\n'+proc.stderr)[-8000:]
    # Summaries, node IDs and exception types only; never put fixture payloads in reports.
    summary=[]
    for line in tail.splitlines():
        if re.search(r'(\d+ passed|\d+ failed|\d+ error|\d+ skipped)',line) or line.startswith(('FAILED tests/','ERROR tests/')):
            summary.append(re.sub(r'[^a-zA-Z0-9_./:\- ()=,\[\]]','',line)[:220])
    if proc.returncode:
        raise AssertionError('pytest failed: '+('; '.join(summary[-8:]) or 'nonzero status'))
    return {'status':'PASS','summary':summary[-6:]}


def _compression(temp):
    from .probe import probe_model
    probe=probe_model(temp, allowed_increase=.05,characters=100)
    assert probe['selected_validation']['cross_entropy'] <= probe['base_validation']['cross_entropy']+.05+1e-6
    assert probe['selected_payload_bytes']<=probe['original_payload_bytes']
    return {'original_bytes':probe['original_payload_bytes'],
            'selected_bytes':probe['selected_payload_bytes'],
            'selected_operator_kinds':[r['selected']['kind'] for r in probe['operators']],
            'validation_loss_delta':round(probe['selected_validation']['cross_entropy']-probe['base_validation']['cross_entropy'],6),
            'sealed_test_loss_delta':round(probe['selected_test']['cross_entropy']-probe['base_test']['cross_entropy'],6),
            'baseline_ms_per_character':round(probe['timing']['baseline_ms_per_character'],5),
            'selected_ms_per_character':round(probe['timing']['selected_ms_per_character'],5)}


def _latency_policy(temp):
    from .probe import probe_model
    check=probe_model(temp,allowed_increase=.05,characters=100,
                      objective='latency_guarded',max_slowdown=.05)
    gate=check['decision']
    assert check['selected_validation']['cross_entropy'] <= check['base_validation']['cross_entropy']+.05+1e-6
    if gate['accepted']:
        assert check['timing']['candidate_to_baseline_ratio'] <= 1.05+1e-8
    else:
        assert gate['reason']=='reverted_to_dense_due_to_validation_latency'
        assert check['selected_payload_bytes']==check['original_payload_bytes']
        assert abs(check['selected_validation']['cross_entropy']-check['base_validation']['cross_entropy'])<1e-8
    return {'accepted_candidate':gate['accepted'],
            'candidate_to_baseline_runtime_ratio':round(check['timing']['candidate_to_baseline_ratio'],3),
            'applied_bytes':check['selected_payload_bytes'],
            'reason':gate['reason']}


def _training(temp):
    from .train import train_model
    from .model import CharacterModel
    dest=temp/'models'/'doctor-training'
    result=train_model(temp/'data'/'corpus.txt',dest,kind='factorized',hidden=24,rank=8,
                       layers=2,steps=35,seed=41,sequence_length=20,batch_size=8)
    m=CharacterModel(dest)
    assert m.layers==2 and result['parity_max_error']<1e-4
    return {'layers':m.layers,'training_steps':35,'parity_max_error':result['parity_max_error'],
            'validation_loss':result['training']['validation_loss']}


def _browser(temp):
    """Optional real-browser smoke test of graph, file import, evidence and probes."""
    import shutil as _shutil
    if not Path('/usr/bin/chromium').is_file():
        return {'status':'SKIP','reason':'chromium_not_found'}
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env={**os.environ,'PYTHONPATH':str(temp),'PYTHONUNBUFFERED':'1',
         'ATLAS_SMOKE_PORT':str(port)}
    proc=subprocess.Popen([sys.executable,'-m','atlas.cli','serve','--port',str(port)],
                          cwd=temp,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if proc.poll() is not None:raise RuntimeError('Browser smoke server could not start')
            try:
                with urlopen(f'http://127.0.0.1:{port}/api/status',timeout=1):break
            except OSError:time.sleep(.1)
        result=subprocess.run([sys.executable,'tests/browser_smoke.py'],cwd=temp,env=env,
                              capture_output=True,text=True,timeout=85)
        if result.returncode:raise AssertionError('Browser smoke failed with exit '+str(result.returncode))
        assert 'desktop PASS' in result.stdout and 'mobile PASS' in result.stdout
        return {'desktop':'PASS','mobile':'PASS'}
    finally:
        proc.terminate()
        try:proc.wait(timeout=4)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)


def run_doctor(root, output=None, quick=False):
    root=Path(root).resolve()
    start=time.perf_counter();checks=[];fixture_payload=None;equation_study=None;evidence_study=None;coordinate_study=None;real_study=None
    with tempfile.TemporaryDirectory(prefix='atlas-doctor-') as td:
        scratch=Path(td)
        _run('release_file_integrity',lambda:_integrity(root),checks)
        _run('isolated_workspace',lambda:_sandbox(root,scratch),checks)
        if checks[-1]['status']=='PASS':
            _run('numerical_operators_and_models',lambda:_numerical(scratch),checks)
            _run('evidence_graph_generation',lambda:_workflows(scratch),checks)
            _run('character_decoding_sensitivity',lambda:_language(scratch),checks)
            _run('public_fixture_ingestion_equation_learning',lambda:_fixture_learning(scratch),checks)
            if checks[-1]['status']=='PASS':
                fixture_payload=checks[-1].pop('measurements')
                checks[-1]['measurements']={'before_route':fixture_payload['before']['response_route'],
                    'after_route':fixture_payload['after']['response_route'],
                    'equation_route':fixture_payload['equation']['response_route'],
                    'before_found':fixture_payload['before']['found'],
                    'after_found':fixture_payload['after']['found'],
                    'equation_validation_rmse':fixture_payload['equation']['validation_rmse'],
                    'equation_error_on_unseen_point':fixture_payload['equation']['absolute_error'],
                    'random_control_rejected':fixture_payload['equation']['random_control_refused'],
                    'out_of_domain_rejected':fixture_payload['equation']['outside_range_refused'],
                    'fixture_count':len(fixture_payload['fixture_hashes']),
                    'composition_supported':fixture_payload['composition']['supported'],
                    'composition_domain_refused':fixture_payload['composition']['outside_range_refused'],
                    'second_source_equation_id':fixture_payload['composition']['second_equation']}
            from .examples import evidence_quality_study
            _run('controlled_document_qa_and_conflicts',lambda:evidence_quality_study(scratch,scratch/'qa-study'),checks)
            if checks[-1]['status']=='PASS':
                evidence_study=checks[-1].pop('measurements')
                checks[-1]['measurements']={k:evidence_study[k] for k in ('positive_total','positive_correct','negative_total','negative_abstained','conflict_reported','naive_chunk_search_false_positive_count')}
            from .evaluation import coordinate_family_study
            _run('coordinate_and_extended_family_study',lambda:coordinate_family_study(),checks)
            if checks[-1]['status']=='PASS':
                coordinate_study=checks[-1].pop('measurements')
                checks[-1]['measurements']={k:coordinate_study[k] for k in ('seeds','mean_unseen_rmse_13','mean_unseen_rmse_16','large_coordinate_max_abs_error')}
            _run('independent_equation_family_study',lambda:_equation_study(scratch),checks)
            if checks[-1]['status']=='PASS':
                equation_study=checks[-1].pop('measurements')
                checks[-1]['measurements']={'seeds':len(equation_study['seeds']),
                    'summaries':equation_study['summary'],
                    'synthetic_only':True}
            from .evaluation import real_world_sunspot_study
            _run('historical_non_synthetic_stress',lambda:real_world_sunspot_study(scratch),checks)
            if checks[-1]['status']=='PASS':
                real_study=checks[-1].pop('measurements')
                checks[-1]['measurements']={'dataset_rows':real_study['rows'],'chronological_splits':len(real_study['runs']),**real_study['averages']}
            _run('local_http_and_origin_policy',lambda:_http(scratch),checks)
            if not quick:
                _run('trained_model_compression',lambda:_compression(scratch),checks)
                _run('latency_guarded_selection',lambda:_latency_policy(scratch),checks)
                if importlib.util.find_spec('torch') is None:
                    checks.append({'name':'optional_training_reexport','status':'SKIP','reason':'torch_not_installed'})
                else:
                    _run('optional_training_reexport',lambda:_training(scratch),checks)
                if importlib.util.find_spec('playwright') is None or not Path('/usr/bin/chromium').is_file():
                    checks.append({'name':'optional_chromium_ui','status':'SKIP','reason':'playwright_or_chromium_missing'})
                else:
                    _run('optional_chromium_ui',lambda:_browser(scratch),checks)
                if importlib.util.find_spec('pytest') is None:
                    checks.append({'name':'full_pytest_suite','status':'SKIP','reason':'pytest_not_installed'})
                else:
                    _run('full_pytest_suite',lambda:_pytest(scratch),checks)
    failed=sum(c['status']=='FAIL' for c in checks)
    skipped=sum(c['status']=='SKIP' for c in checks)
    payload={'schema':'operator-atlas-doctor-v1','version':'1.0.0','mode':'quick' if quick else 'full',
             'result':'FAIL' if failed else 'PARTIAL' if skipped else 'PASS',
             'checks':checks,'counts':{'passed':sum(c['status']=='PASS' for c in checks),'failed':failed,'skipped':skipped},
             'elapsed_seconds':round(time.perf_counter()-start,3),
             'isolation':'All active tests run in a disposable copy without user database, uploaded documents, logs, or training candidates',
             'scientific_limit':'Tests validate this implementation, not language ability or superiority over compact dense models'}
    # Extend the existing diagnostic ZIP format; it remains the one report owner.
    from .diagnostics import EventLog
    EventLog(root/'logs'/'events.jsonl').record('doctor.error' if failed else 'doctor.ok',
        action='doctor',result=payload['result'].lower(),elapsed_ms=round((time.perf_counter()-start)*1000,2))
    attachments={'doctor.json':payload}
    if fixture_payload: attachments['fixture_demo.json']=fixture_payload
    if equation_study: attachments['equation_study.json']=equation_study
    if evidence_study: attachments['evidence_quality_study.json']=evidence_study
    if coordinate_study: attachments['coordinate_study.json']=coordinate_study
    if real_study: attachments['historical_sunspots.json']=real_study
    report=export_report(root,output,attachments=attachments)
    return {'result':payload['result'],'checks':payload['counts'],'report_path':report['report_path'],
            'files':report['files'],'elapsed_seconds':payload['elapsed_seconds']}
