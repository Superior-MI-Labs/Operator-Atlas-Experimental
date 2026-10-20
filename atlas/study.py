"""Matched-budget stacked RNN study using the canonical trainer and checkpoint reader."""
from pathlib import Path
import hashlib
from statistics import mean, stdev
from .train import train_model
from .model import CharacterModel
from .diagnostics import EventLog


def run_stacked_study(root, steps=300, seeds=(3,7,11)):
    root=Path(root)
    steps=int(steps)
    if not 30<=steps<=1500:raise ValueError('Training steps must be 30..1500')
    seeds=tuple(int(s) for s in seeds)
    if not (2<=len(seeds)<=8 and len(set(seeds))==len(seeds)):
        raise ValueError('Use 2 to 8 distinct seeds')
    rows=[]
    for seed in seeds:
        for kind,hidden in (('dense',38),('factorized',48)):
            dest=root/'models'/'candidates'/f'stacked_{kind}_initvariance_seed{seed}_steps{steps}'
            if dest.exists():
                model=CharacterModel(dest)
                training=model.manifest['training']
                digest=hashlib.sha256((root/'data'/'corpus.txt').read_bytes()).hexdigest()
                if (training.get('corpus_sha256')!=digest or training.get('seed')!=seed or
                    training.get('steps')!=steps or training.get('initializer')!='variance_matched_v1' or training.get('layers')!=2 or model.hidden!=hidden or
                    ('factorized' if kind=='factorized' else 'dense')!=model.operators['input'].kind):
                    raise ValueError('Existing checkpoint disagrees with experiment specification')
                info={'checkpoint_id':model.manifest['checkpoint_id'],'training':training,
                      'parameter_bytes':model.status()['parameter_bytes'],
                      'parity_max_error':training['numpy_torch_max_error']}
            else:
                info=train_model(root/'data'/'corpus.txt',dest,kind=kind,hidden=hidden,rank=16,
                                 layers=2,steps=steps,seed=seed)
                model=CharacterModel(dest)
            # This is a reporting-only holdout; it does not influence training or architecture selection.
            text=(root/'data'/'corpus.txt').read_text()
            test=text[int(len(text)*.94):]
            nll=model.score_text(test[:1000])['cross_entropy']
            rows.append({'seed':seed,'kind':kind,'layers':2,'hidden':hidden,
                         'checkpoint':info['checkpoint_id'], 'stored_bytes':info['parameter_bytes'],
                         'validation_cross_entropy':info['training']['validation_loss'],
                         'reporting_test_cross_entropy':nll,'training_seconds':info['training']['seconds'],
                         'parity_error':info['parity_max_error']})
    summary={}
    for kind in ('dense','factorized'):
        group=[r for r in rows if r['kind']==kind]
        loss=[r['reporting_test_cross_entropy'] for r in group]
        summary[kind]={'mean_test_loss':mean(loss),'sd_test_loss':stdev(loss) if len(loss)>1 else None,
                       'stored_bytes':group[0]['stored_bytes'],'seeds':len(group)}
    record={'schema':1,'experiment':'two_layer_matched_budget','steps':steps,'seeds':list(seeds),
            'comparison':'same training data and optimizer; factorized width48 rank16 vs dense width38',
            'summary':summary,'runs':rows,
            'limitations':'Authored repeated corpus, character prediction, not independent language understanding; no proof of scaling or quality benefit.'}
    eid=EventLog(root/'logs'/'events.jsonl').save_experiment(root/'results'/'runs','study',
                  {'steps':steps,'seeds':list(seeds)},record)
    return {'run_id':eid,**record}
