"""Controlled multi-seed training comparison, same data/optimizer and near-equal byte budgets."""
import json
from pathlib import Path
from .train import train_model

def run(root,seeds=(7,11,19),steps=700):
    root=Path(root);rows=[]
    for seed in seeds:
        for kind,hidden,rank,label in [('factorized',48,16,'factorized_rank16'),('dense',38,16,'dense_width38')]:
            out=root/'models'/'ablations'/f'{label}_seed{seed}'
            result=train_model(root/'data'/'corpus.txt',out,kind=kind,hidden=hidden,rank=rank,steps=steps,seed=seed)
            rows.append({'seed':seed,'variant':label,'kind':kind,'hidden':hidden,'rank':rank if kind=='factorized' else None,
                         'bytes':result['parameter_bytes'],'validation_loss':result['training']['validation_loss'],
                         'train_loss':result['training']['train_loss'],'training_seconds':result['training']['seconds'],
                         'inference_parity_max_error':result['parity_max_error'],'checkpoint':result['checkpoint_id']})
            print(f"{label} seed {seed}: bytes={rows[-1]['bytes']} loss={rows[-1]['validation_loss']:.5f}",flush=True)
    from statistics import mean
    summary={label:{'mean_val_loss':mean(r['validation_loss'] for r in rows if r['variant']==label),
                    'mean_payload_bytes':mean(r['bytes'] for r in rows if r['variant']==label)}
             for label in ('factorized_rank16','dense_width38')}
    payload={'schema':1,'task':'authored templated character corpus','seeds':list(seeds),
             'comparison':'Near-equal byte budgets, not equal effective training FLOPs. Same dataset, optimizer family, sequence/batch sizes and step counts.',
             'runs':rows,'summary':summary}
    (root/'results'/'multi_seed_ablation.json').write_text(json.dumps(payload,indent=2))
    return payload
