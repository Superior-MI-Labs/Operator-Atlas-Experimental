"""Local HTTP server; no user-facing GUI owns model state.

No CDN, no external network requests, no remote bind by default.
"""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
from pathlib import Path
import base64,json,mimetypes,time
from .model import CharacterModel
from .memory import EvidenceStore
from .discovery import discover,predict
from .evaluation import evaluate
from .graph import graph
from .diagnostics import EventLog
from .importers import decode_document,MAX_FILE_BYTES
from .probe import probe_model
from .responder import respond

class AtlasService:
    def __init__(self,root,model_dir=None):
        self.root=Path(root).resolve()
        self.db=EvidenceStore(self.root/'data'/'evidence.sqlite3')
        self.events=EventLog(self.root/'logs'/'events.jsonl')
        self.model_dir=Path(model_dir).resolve() if model_dir else self.root/'models'/'starter'
        self.model=CharacterModel(self.model_dir) if (self.model_dir/'manifest.json').exists() else None
    def status(self):
        ablation_path=self.root/'results'/'multi_seed_ablation.json'
        ablation=json.loads(ablation_path.read_text())['summary'] if ablation_path.exists() else None
        stacked_path=self.root/'results'/'stacked_ablation_v04.json'
        stacked=json.loads(stacked_path.read_text()).get('summary') if stacked_path.exists() else None
        research_files=('evidence_quality_study_v09.json','coordinate_study_v09.json')
        recent_research={}
        for name in research_files:
            path=self.root/'results'/name
            if path.is_file():
                record=json.loads(path.read_text())
                if name.startswith('evidence_'):
                    recent_research['source_answers']={key:record.get(key) for key in
                        ('positive_total','positive_correct','negative_total','negative_abstained','conflict_reported','naive_chunk_search_false_positive_count')}
                else:
                    recent_research['mathematical_coordinates']={key:record.get(key) for key in
                        ('mean_unseen_rmse_13','mean_unseen_rmse_16','large_coordinate_max_abs_error')}
        historical_path=self.root/'results'/'historical_sunspots_v1.json'
        historical_summary=json.loads(historical_path.read_text())['averages'] if historical_path.is_file() else None
        return {'historical_sunspot_stress':historical_summary,'research_v09':recent_research,'ablation_summary':ablation,'stacked_study_summary':stacked,'system':'Superior MI Labs — Operator Atlas','version':'1.0.0','model_loaded':bool(self.model),
            'model':self.model.status() if self.model else None,
            'dense_control':CharacterModel(self.root/'models'/'dense-control').status() if (self.root/'models'/'dense-control'/'manifest.json').exists() else None,
            'evidence':self.db.stats(),
            'discovery':'16-term positive-domain grammar / 13-term general domain, coordinate-aware polynomial terms, BIC and held-out diagnostics',
            'bounds':'Experimental character model, not a general conversational AI; equation compositions do not verify unit compatibility',
            'ingestion_formats':['txt','md','csv','json','html','pdf (optional pypdf)'],
            'diagnostic_log':'local sanitized JSONL; raw user text is not logged'}
    def process(self, action, data):
        """Single service action path shared by CLI and local HTTP UI."""
        run_id=self.events.new_id()
        started=time.perf_counter()
        try:
            output=self._execute(action,data)
            fields={'action':action,'request_id':run_id,'elapsed_ms':round((time.perf_counter()-started)*1000,3),
                    'result':'success'}
            if action in ('ingest','ingest_file','load_example'):fields['source_id']=output['source_id']
            if action=='generate':fields['checkpoint']=self.model.manifest.get('checkpoint_id')
            if action=='respond':fields['route']=output['route']
            if action=='search':fields['documents']=len(output['results'])
            if action=='learn_equation':fields['source_id']=output['source_id']
            if action=='probe':fields['candidates']=sum(len(r['alternatives']) for r in output['operators'])
            if action in ('experiments','probe'):
                output['run_id']=self.events.save_experiment(self.root/'results'/'runs',action,
                    {'seed':int(data.get('seed',41))} if action=='experiments' else
                    {'characters':int(data.get('characters',400)),'allowed_increase':float(data.get('allowed_increase',.05)),
                     'objective':str(data.get('objective','storage')),'max_slowdown':float(data.get('max_slowdown',.05))},output)
            self.events.record('request.ok',**fields)
            return output
        except Exception as exc:
            self.events.record('request.error',action=action,request_id=run_id,
                               elapsed_ms=round((time.perf_counter()-started)*1000,3),
                               reason=type(exc).__name__)
            raise

    def _execute(self,action,data):
        if action=='respond':return respond(self,str(data.get('question','')),str(data.get('mode','grounded')),
                                            data.get('max_new',65),data.get('temperature',.65),data.get('seed',12))
        if action=='generate':
            if not self.model:raise ValueError('No trained model. Run ./launch.sh train first.')
            return self.model.generate(str(data.get('prompt',''))[:500],data.get('max_new',65),data.get('temperature',.75),data.get('seed',12))
        if action=='fit_equation':
            pts=data.get('points',[])
            fitted=discover(pts)
            xs=[p[0] for p in pts[:80]]
            fitted['predictions']=predict(fitted,xs)
            return fitted
        if action=='load_example':
            # Explicit user action; whitelist avoids arbitrary server-side paths.
            from .examples import FIXTURES
            name=str(data.get('name',''))
            if name not in FIXTURES:raise ValueError('Unknown packaged demonstration fixture')
            text,digest,mime=decode_document(name,(self.root/'examples'/'fixtures'/name).read_bytes())
            return {**self.db.ingest(name,text,source_name=name,media_type=mime,original_sha256=digest),
                    'name':name,'fictional':True}
        if action=='ingest':return self.db.ingest(data.get('title',''),data.get('text',''))
        if action=='ingest_file':
            filename=str(data.get('filename',''))
            raw=str(data.get('base64',''))
            if len(raw)>int(MAX_FILE_BYTES*4/3)+8:raise ValueError('File exceeds 5 MB limit')
            try:content=base64.b64decode(raw,validate=True)
            except Exception as exc:raise ValueError('Invalid file encoding') from exc
            text,digest,mime=decode_document(filename,content)
            from pathlib import Path
            result=self.db.ingest(Path(filename).name,text,source_name=Path(filename).name,
                                  media_type=mime,original_sha256=digest)
            return {**result,'filename':Path(filename).name,'original_sha256':digest,'characters':len(text)}
        if action=='sources':return {'sources':self.db.list_sources(data.get('limit',40))}
        if action=='learn_equation':return self.db.learn_equation(data.get('source_id'),data.get('x_column'),data.get('y_column'))
        if action=='equations':return {'equations':self.db.list_equations(data.get('limit',40))}
        if action=='predict_equation':return self.db.predict_equation(data.get('equation_id'),data.get('x'))
        if action=='compose_equations':return self.db.compose_equations(data.get('equation_ids'),data.get('x'))
        if action=='search':return {'results':self.db.search(data.get('query',''),data.get('limit',5))}
        if action=='ask_sources':return self.db.extractive_answer(data.get('question',''),data.get('limit',3))
        if action=='experiments':return evaluate(int(data.get('seed',41)))
        if action=='probe':return probe_model(self.root,self.root/'models'/'dense-control',
                         data.get('allowed_increase',.05),data.get('characters',400),
                         objective=data.get('objective','storage'),max_slowdown=data.get('max_slowdown',.05))
        raise ValueError('Unknown action')

def serve(root,port=8765,host='127.0.0.1',model_dir=None):
    if host not in ('127.0.0.1','localhost'):
        raise ValueError('Research server can bind to loopback only; public hosting requires authenticated isolation')
    if host not in ('127.0.0.1','localhost','::1'):
        raise ValueError('Remote binding disabled: local research server is not authenticated')
    service=AtlasService(root,model_dir);webroot=(service.root/'web').resolve()
    service.events.record('service.start',checkpoint=service.model.manifest.get('checkpoint_id') if service.model else None)
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,data):
            raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/status':return self.respond(200,service.status())
            if path=='/api/graph':return self.respond(200,graph(service.model))
            if path=='/api/events':return self.respond(200,{'events':service.events.tail(100), 'privacy':'No prompts, source text or file paths are recorded'})
            if path.startswith('/api/operator/'):
                if not service.model:return self.respond(404,{'error':'no checkpoint'})
                name=path.split('/')[-1];op=service.model.operators.get(name)
                if not op:return self.respond(404,{'error':'unknown operator'})
                return self.respond(200,{'spec':op.spec(),'heatmap':op.heatmap()})
            if path=='/':path='/index.html'
            if path not in ('/index.html','/app.js','/style.css'):
                self.send_error(404);return
            file=(webroot/path.lstrip('/')).resolve()
            if not file.is_file() or webroot not in file.parents:self.send_error(404);return
            contents=file.read_bytes();self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(len(contents)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(contents)
        def do_POST(self):
            path=urlparse(self.path).path
            # Block cross-site JSON POSTs from unrelated websites; CLI calls omit Origin.
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                return self.respond(403,{'error':'cross-origin requests blocked'})
            if not path.startswith('/api/'):
                return self.respond(404,{'error':'unknown route'})
            try:
                sz=int(self.headers.get('Content-Length','0'))
                max_request=7_000_000 if path=='/api/ingest_file' else 1_700_000 if path=='/api/ingest' else 250_000
                if not (0<sz<=max_request):return self.respond(413,{'error':'JSON request exceeds local action size limit'})
                data=json.loads(self.rfile.read(sz))
                if not isinstance(data,dict):raise ValueError('Expected JSON object')
                return self.respond(200,service.process(path.removeprefix('/api/'),data))
            except (ValueError,TypeError,KeyError) as exc:return self.respond(400,{'error':str(exc)})
            except Exception as exc:
                # No tracebacks delivered to browser; no user-provided stack echoes.
                print('Request failed:',type(exc).__name__);return self.respond(500,{'error':'Internal execution failed; inspect server log'})
        def log_message(self,fmt,*args):pass
    server=ThreadingHTTPServer((host,int(port)),Handler)
    print(f'Operator Atlas → http://{host}:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
