import base64
import json
from pathlib import Path
import pytest
from atlas.api import AtlasService
from atlas.importers import decode_document
from atlas.diagnostics import EventLog
from atlas.memory import EvidenceStore

ROOT=Path(__file__).resolve().parent.parent

def test_document_import_content_hash_and_citations(tmp_path):
    service=AtlasService(tmp_path)
    content=b'Copper conducts electricity. The Atlas design retains provenance and refuses to invent new scientific facts.'
    data={'filename':'notes.md','base64':base64.b64encode(content).decode()}
    result=service.process('ingest_file',data)
    assert result['original_sha256'] and result['sha256']
    assert service.process('ingest_file',data)['source_id']==result['source_id']
    hits=service.process('ask_sources',{'question':'copper electricity'})
    assert hits['found'] and hits['passages'][0]['source_id']==result['source_id']
    sources=service.process('sources',{})['sources']
    assert len(sources)==1 and sources[0]['source_name']=='notes.md'
    assert sources[0]['original_sha256']==result['original_sha256']
    assert service.process('ask_sources',{'question':'unrelatedgalaxyword'})['found'] is False

def test_existing_sqlite_schema_migrates_without_data_loss(tmp_path):
    import sqlite3
    path=tmp_path/'legacy.sqlite3'
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE sources (id INTEGER PRIMARY KEY, title TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, content TEXT NOT NULL, created_at REAL NOT NULL)')
        con.execute("INSERT INTO sources VALUES(1,'old','abc','existing archived source',1)")
    store=EvidenceStore(path)
    assert store.list_sources()[0]['title']=='old'
    store.ingest('new','A new document holding enough characters for storage')
    assert store.stats()['source_count']==2

def test_format_validation_and_html_safety():
    text,digest,mime=decode_document('note.html',b'<p>Visible mathematical text and facts go here.</p><script>secretXYZ();</script>')
    assert 'secretXYZ' not in text and 'Visible' in text and mime=='text/html'
    with pytest.raises(ValueError,match='Unsupported'):
        decode_document('payload.py',b'A large code file that is not a supported document')
    with pytest.raises(ValueError,match='UTF-8'):
        decode_document('bad.txt',b'\xff'*40)
    with pytest.raises(ValueError,match='Invalid JSON'):
        decode_document('bad.json',b'{not a valid JSON document at all}')
    with pytest.raises(ValueError):
        decode_document('huge.txt',b'a'*5_000_001)

def test_event_log_is_sanitized_and_experiment_is_immutable(tmp_path):
    e=EventLog(tmp_path/'logs/events.jsonl')
    secret='PRIVATE_BOOK_CONTENT_24680'
    e.record('request.ok',action='search',result='success',prompt=secret,document=secret)
    assert secret not in e.path.read_text()
    assert 'prompt' not in e.tail()[0]
    run_id=e.save_experiment(tmp_path/'results/runs','probe',{'characters':100},{'cross_entropy':.2})
    assert (tmp_path/'results/runs'/f'{run_id}.json').exists()
    assert json.loads((tmp_path/'results/runs'/f'{run_id}.json').read_text())['result']['cross_entropy']==.2
    assert e.tail()[-1]['event']=='experiment.saved'
    with pytest.raises(ValueError):e.record('arbitrary event',raw=secret)

def test_service_observes_failures_without_exposing_input(tmp_path):
    service=AtlasService(tmp_path)
    with pytest.raises(ValueError):service.process('ingest_file',{'filename':'s.txt','base64':'private not base64'})
    events=service.events.tail()
    assert events[-1]['event']=='request.error'
    assert 'private not base64' not in service.events.path.read_text()
    assert events[-1]['reason']=='ValueError'

def test_launcher_prefers_venv_and_supports_import_probe():
    script=(ROOT/'launch.sh').read_text()
    assert '.venv/bin/python' in script
    assert '"$ATLAS_PYTHON" -m atlas.cli' in script
    assert 'import-file' in script and 'probe' in script and 'logs' in script

def test_explicit_offline_migration_preserves_ids_and_refuses_overwrite(tmp_path):
    old=EvidenceStore(tmp_path/'old.sqlite3')
    a=old.ingest('Original', 'The original archival statement contains distinctive language.')
    new=EvidenceStore(tmp_path/'new.sqlite3')
    assert new.migrate_from(old.path)['source_count']==1
    assert new.list_sources()[0]['source_id']==a['source_id']
    with pytest.raises(ValueError,match='already contains'):
        new.migrate_from(old.path)

def test_support_report_excludes_source_text_and_sqlite(tmp_path):
    import zipfile
    from atlas.diagnostics import export_report
    from atlas.memory import EvidenceStore
    (tmp_path/'data').mkdir()
    EvidenceStore(tmp_path/'data/evidence.sqlite3').ingest('secret', 'UNIQUE_PRIVATE_SOURCE_3141592 is never exported in the support zip')
    result=export_report(tmp_path,tmp_path/'report.zip')
    with zipfile.ZipFile(result['report_path']) as z:
        names=z.namelist()
        contents=''.join(z.read(name).decode() for name in names)
    assert names==['report.json','events.json','experiment_index.json']
    assert 'UNIQUE_PRIVATE_SOURCE_3141592' not in contents
    assert 'evidence.sqlite' not in str(names)
