"""Single local JSONL research event authority. No prompt/source text in records.

Events are for debugging and reproducibility, not an audit trail of user content.
"""
import json
import os
import threading
import time
import shutil
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_ALLOWED = {'service.start', 'service.error', 'request.ok', 'request.error',
            'training.ok', 'training.error', 'experiment.saved', 'file.rejected', 'doctor.ok', 'doctor.error'}

class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event, **fields):
        if event not in _ALLOWED:
            raise ValueError('Unrecognized diagnostic event')
        # Explicit schema: callers cannot accidentally persist text, prompts or file paths.
        allowed = ('action', 'elapsed_ms', 'result', 'request_id', 'checkpoint',
                   'model_kind', 'seed', 'input_bytes', 'output_bytes', 'source_id',
                   'documents', 'candidates', 'reason', 'saved_id')
        safe = {k: v for k, v in fields.items() if k in allowed and (v is None or isinstance(v, (str, int, float, bool)))}
        # Request paths are not a license to log arbitrary user-chosen strings.
        import re
        for key in ('action','result','reason','model_kind'):
            value = safe.get(key)
            if isinstance(value,str) and not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',value):
                safe[key]='redacted'
        raw = json.dumps({'schema': 1, 'time': datetime.now(timezone.utc).isoformat(),
                          'event': event, **safe}, allow_nan=False, separators=(',', ':')) + '\n'
        with self._lock:
            # O_APPEND protects independent writers from overwriting existing events.
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.write(fd, raw.encode('utf-8'))
            finally:
                os.close(fd)

    def tail(self, limit=60):
        limit = max(1, min(int(limit), 200))
        if not self.path.exists():
            return []
        with self._lock, self.path.open('r', encoding='utf-8') as f:
            return [json.loads(s) for s in deque(f, maxlen=limit) if s.strip()]

    def save_experiment(self, directory, action, parameters, result):
        """Persist result atomically without mutable latest.json pointers."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        run_id = uuid4().hex[:16]
        payload = {'schema': 1, 'run_id': run_id, 'created_at': datetime.now(timezone.utc).isoformat(),
                   'action': action, 'parameters': parameters, 'result': result}
        raw = json.dumps(payload, allow_nan=False, indent=2, ensure_ascii=False).encode('utf-8')
        dest = directory / (run_id + '.json')
        temp = directory / ('.' + run_id + '.tmp')
        with temp.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, dest)
        self.record('experiment.saved', action=action, saved_id=run_id, output_bytes=len(raw))
        return run_id

    @staticmethod
    def new_id():
        return uuid4().hex[:12]

def export_report(root, output=None, attachments=None):
    """Build a shareable diagnostic package without evidence text or user prompts.

    Callers should still review it before publishing: checkpoint and host metadata
    may reveal environment details. Reports never include SQLite or source files.
    """
    import platform
    import sys
    import zipfile
    import numpy as np
    root=Path(root)
    output=Path(output).expanduser() if output else Path.home()/'Downloads'/('Operator-Atlas-Diagnostic-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'.zip')
    output.parent.mkdir(parents=True,exist_ok=True)
    from .api import AtlasService
    try:
        service=AtlasService(root)
        status=service.status()
        events=service.events.tail(200)
    except Exception as exc:
        # A damaged installation must still produce an actionable report.
        service=None
        status={'status':'failed_to_read_runtime','error_type':type(exc).__name__}
        events=[]
    machine={'python':sys.version.split()[0],'numpy':np.__version__,
             'os':platform.system(),'os_release':platform.release(),
             'architecture':platform.machine(),'cpu_count':os.cpu_count()}
    # Optional hardware capabilities, never GPU UUID, serial number, process list,
    # paths, or board topology. This is not a benchmark or CUDA readiness proof.
    nvidia=shutil.which('nvidia-smi')
    if nvidia:
        try:
            found=subprocess.run([nvidia,'--query-gpu=name,memory.total,driver_version',
                                  '--format=csv,noheader'],capture_output=True,text=True,
                                 timeout=3,check=True)
            gpu=[]
            for line in found.stdout.splitlines()[:8]:
                parts=[part.strip()[:96] for part in line.split(',')]
                if len(parts)==3:gpu.append({'name':parts[0],
                     'memory_total':parts[1], 'driver_version':parts[2]})
            machine['nvidia_gpus']=gpu
        except (OSError,subprocess.TimeoutExpired,subprocess.CalledProcessError):
            machine['nvidia_probe']='unavailable'
    recent=[]
    for path in sorted((root/'results'/'runs').glob('*.json'))[-15:]:
        try:
            record=json.loads(path.read_text())
            if record.get('action') not in ('probe','experiments','study'):continue
            recent.append({'run_id':record.get('run_id'),'action':record.get('action'),
                           'created_at':record.get('created_at'),
                           'parameters':record.get('parameters'),
                           'result_summary':{
                                'selected_payload_bytes':record.get('result',{}).get('selected_payload_bytes'),
                                'original_payload_bytes':record.get('result',{}).get('original_payload_bytes'),
                                'selected_validation':record.get('result',{}).get('selected_validation'),
                                'selected_test':record.get('result',{}).get('selected_test'),
                                'seed':record.get('result',{}).get('seed'),
                                'study_summary':record.get('result',{}).get('summary')}})
        except (ValueError,OSError):continue
    contents={'report.json':{'format':'operator-atlas-diagnostics-v1','generated':datetime.now(timezone.utc).isoformat(),
                             'system':status,'machine':machine,
                             'privacy':'No uploaded file contents, source SQLite DB or user prompts included; review before sharing.'},
              'events.json':events,
              'experiment_index.json':recent}
    if attachments:
        for name, payload in attachments.items():
            if name not in ('doctor.json','fixture_demo.json','equation_study.json','evidence_quality_study.json','coordinate_study.json','historical_sunspots.json'):
                raise ValueError('Unsupported report attachment')
            contents[name]=payload
    tmp=output.with_suffix(output.suffix+'.tmp')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for name,payload in contents.items():
                z.writestr(name,json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
        os.replace(tmp,output)
    finally:
        if tmp.exists():tmp.unlink()
    return {'report_path':str(output),'files':list(contents),'contains_private_documents':False}
