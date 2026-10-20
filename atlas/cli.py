"""CLI: one backend for all graphical and scripted access."""
import argparse,json,sys
from pathlib import Path
from . import __version__
ROOT=Path(__file__).resolve().parent.parent

def main(argv=None):
    p=argparse.ArgumentParser(description='Superior MI Labs | Operator Atlas v'+__version__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('serve',help='launch local visual graph and API');s.add_argument('--port',type=int,default=8765);s.add_argument('--host',default='127.0.0.1');s.add_argument('--model',help='path to an exported candidate checkpoint')
    t=sub.add_parser('train',help='train real small autoregressive character model');t.add_argument('--kind',choices=['dense','factorized'],default='factorized');t.add_argument('--steps',type=int,default=220);t.add_argument('--seed',type=int,default=7);t.add_argument('--hidden',type=int,default=48);t.add_argument('--rank',type=int,default=16);t.add_argument('--layers',type=int,default=1);t.add_argument('--out',default=None,help='checkpoint destination; default is a candidate directory')
    q=sub.add_parser('generate',help='run NumPy inference');q.add_argument('prompt');q.add_argument('--tokens',type=int,default=80);q.add_argument('--seed',type=int,default=12);q.add_argument('--model',help='checkpoint path')
    sub.add_parser('simulate',help='run reproducible numerical and timing experiments')
    sub.add_parser('study-real',help='run predeclared historical sunspot stress study; saves public JSON')
    demo=sub.add_parser('demonstrate',help='run shipped fixture before/after workflow without touching personal evidence')
    demo.add_argument('--out',default=None,help='write public reproducible demonstration JSON')
    sub.add_parser('diagnose',help='check environment and deployed model')
    f=sub.add_parser('import-file',help='preserve source text from local file (PDF needs optional pypdf)');f.add_argument('file')
    reply=sub.add_parser('respond',help='grounded responses with provenance, or explicit experimental neural continuation');reply.add_argument('question');reply.add_argument('--mode',choices=['grounded','experimental'],default='grounded')
    a=sub.add_parser('ask',help='retrieve source-backed passages, not neural hallucinated answers');a.add_argument('question')
    learn=sub.add_parser('learn-csv',help='fit bounded equation from an already imported numeric CSV source');learn.add_argument('source_id',type=int);learn.add_argument('x_column');learn.add_argument('y_column')
    pred=sub.add_parser('predict-equation',help='evaluate qualified CSV equation within observed range');pred.add_argument('equation_id',type=int);pred.add_argument('x',type=float)
    comp=sub.add_parser('compose-equations',help='evaluate a provenance-linked ordered equation chain')
    comp.add_argument('equation_ids',nargs='+',type=int);comp.add_argument('--at',type=float,required=True)
    m=sub.add_parser('migrate-from',help='copy v0.2 SQLite evidence into empty v0.3 database; stop servers first');m.add_argument('directory')
    st=sub.add_parser('study',help='train two-layer dense and factorized controls across fixed seeds (requires torch)')
    st.add_argument('--steps',type=int,default=300)
    st.add_argument('--seeds',nargs='+',type=int,default=[3,7,11])
    d=sub.add_parser('doctor',help='run isolated numerical, integration and regression tests; always export diagnostic ZIP')
    d.add_argument('--quick',action='store_true',help='skip heavier model probe and optional pytest suite')
    d.add_argument('--out',default=None,help='diagnostic ZIP destination')
    r=sub.add_parser('report',help='create a sanitized ZIP diagnostics bundle for bug reports');r.add_argument('--out',default=None)
    l=sub.add_parser('logs',help='inspect sanitized local diagnostic events');l.add_argument('--limit',type=int,default=40)
    o=sub.add_parser('probe',help='validation-only operator compression experiment on trained checkpoint')
    o.add_argument('--characters',type=int,default=400);o.add_argument('--tolerance',type=float,default=.05)
    o.add_argument('--objective',choices=['storage','latency_guarded'],default='storage')
    o.add_argument('--max-slowdown',type=float,default=.05,help='allowed slowdown fraction for latency-guarded selection')
    o.add_argument('--write-candidate',action='store_true',help='export an immutable testable candidate without promoting it')
    args=p.parse_args(argv)
    if args.command=='serve':
        from .api import serve
        serve(ROOT,args.port,args.host,args.model)
    elif args.command=='train':
        from .train import train_model
        dest=args.out or str(ROOT/'models'/'candidates'/f'{args.kind}_layers{args.layers}_hidden{args.hidden}_seed{args.seed}_steps{args.steps}')
        from .diagnostics import EventLog
        eventlog=EventLog(ROOT/'logs'/'events.jsonl')
        try:
            result=train_model(ROOT/'data'/'corpus.txt',dest,kind=args.kind,hidden=args.hidden,rank=args.rank,steps=args.steps,seed=args.seed,layers=args.layers)
            eventlog.record('training.ok',action='train',checkpoint=result['checkpoint_id'],model_kind=args.kind,seed=args.seed)
        except Exception as exc:
            eventlog.record('training.error',action='train',reason=type(exc).__name__)
            raise
        print(json.dumps(result,indent=2))
    elif args.command=='generate':
        from .model import CharacterModel
        print(json.dumps(CharacterModel(args.model or ROOT/'models'/'starter').generate(args.prompt,args.tokens,seed=args.seed),indent=2,ensure_ascii=False))
    elif args.command=='demonstrate':
        from .examples import fixture_demonstration
        import tempfile
        with tempfile.TemporaryDirectory(prefix='atlas-demo-') as temp:
            result=fixture_demonstration(ROOT,Path(temp))
        if args.out:
            Path(args.out).expanduser().write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(result,indent=2,ensure_ascii=False))
    elif args.command=='import-file':
        from .api import AtlasService
        from .importers import decode_document,MAX_FILE_BYTES
        path=Path(args.file).expanduser()
        if path.stat().st_size>MAX_FILE_BYTES:raise ValueError('File must be <= 5 MB')
        text,digest,mime=decode_document(path.name,path.read_bytes())
        service=AtlasService(ROOT)
        result=service.db.ingest(path.name,text,source_name=path.name,media_type=mime,original_sha256=digest)
        service.events.record('request.ok',action='import-file',source_id=result['source_id'],input_bytes=path.stat().st_size)
        print(json.dumps(result,indent=2))
    elif args.command in ('respond','ask','learn-csv','predict-equation','compose-equations'):
        from .api import AtlasService
        action,data={'respond':('respond',{'question':getattr(args,'question',None),'mode':getattr(args,'mode','grounded')}),
            'ask':('ask_sources',{'question':getattr(args,'question',None)}),
            'learn-csv':('learn_equation',{'source_id':getattr(args,'source_id',None),'x_column':getattr(args,'x_column',None),'y_column':getattr(args,'y_column',None)}),
            'predict-equation':('predict_equation',{'equation_id':getattr(args,'equation_id',None),'x':getattr(args,'x',None)}),
            'compose-equations':('compose_equations',{'equation_ids':getattr(args,'equation_ids',None),'x':getattr(args,'at',None)})}[args.command]
        print(json.dumps(AtlasService(ROOT).process(action,data),indent=2,ensure_ascii=False))
    elif args.command=='migrate-from':
        from .memory import EvidenceStore
        src=Path(args.directory).expanduser()/'data'/'evidence.sqlite3'
        target=EvidenceStore(ROOT/'data'/'evidence.sqlite3')
        print(json.dumps({'source':str(src),'destination':'data/evidence.sqlite3','statistics':target.migrate_from(src)},indent=2))
    elif args.command=='study':
        from .study import run_stacked_study
        print(json.dumps(run_stacked_study(ROOT,args.steps,args.seeds),indent=2))
    elif args.command=='doctor':
        from .doctor import run_doctor
        result=run_doctor(ROOT,args.out,quick=args.quick)
        print(json.dumps(result,indent=2))
        if result['result']=='FAIL':sys.exit(1)
    elif args.command=='report':
        from .diagnostics import export_report
        print(json.dumps(export_report(ROOT,args.out),indent=2))
    elif args.command=='logs':
        from .diagnostics import EventLog
        print(json.dumps(EventLog(ROOT/'logs'/'events.jsonl').tail(args.limit),indent=2))
    elif args.command=='probe':
        from .api import AtlasService
        from .probe import probe_model
        service=AtlasService(ROOT)
        if args.write_candidate:
            result=probe_model(ROOT,allowed_increase=args.tolerance,characters=args.characters,write_candidate=True,objective=args.objective,max_slowdown=args.max_slowdown)
        else:
            result=service.process('probe',{'characters':args.characters,'allowed_increase':args.tolerance,'objective':args.objective,'max_slowdown':args.max_slowdown})
        print(json.dumps(result,indent=2))
    elif args.command=='study-real':
        from .evaluation import real_world_sunspot_study
        record=real_world_sunspot_study(ROOT)
        target=ROOT/'results'/'historical_sunspots_v1.json'
        target.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'saved':str(target),'averages':record['averages'],'disclosure':record['note']},indent=2))
    elif args.command=='simulate':
        from .evaluation import write_results
        r=write_results(ROOT/'results'/'simulation.json');print(json.dumps({'saved':'results/simulation.json','timings':r['timings'],'equation':r['equation']['equation']},indent=2))
    elif args.command=='diagnose':
        from .api import AtlasService
        from .graph import graph
        print(json.dumps({'version':__version__,'root':str(ROOT),'graph_nodes':len(graph()['nodes']),**AtlasService(ROOT).status()},indent=2))

if __name__=='__main__':main()
