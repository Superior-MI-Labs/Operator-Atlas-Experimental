"""Task-level, validation-gated search on the existing trained neural checkpoint.

Not autonomous architecture invention. Test partition NEVER used to select operators.
"""
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from .model import CharacterModel
from .operators import Operator, factorize, quantize


def _candidate_ops(name, op):
    w=op.dense_equivalent()
    variants=[Operator(name,'dense',w.shape,{'w':w}),quantize(name,w)]
    for r in (2,4,8,12,16,24):
        if r>=min(w.shape):continue
        variants.append(factorize(name,w,r))
    return variants


def probe_model(root, model_dir=None, allowed_increase=.05, characters=400, write_candidate=False,
                objective="storage", max_slowdown=.05):
    """Search operator substitutions by held-out *task* loss, not weight error.

    The corpus's last 12% was withheld from original training; here it is
    halved again: first part controls selection, second is sealed reporting.
    Repetitions in the authored corpus limit scientific conclusions.
    """
    root=Path(root)
    if objective not in ('storage','latency_guarded'):
        raise ValueError('Unknown probe objective')
    if not 0<=float(max_slowdown)<=1:
        raise ValueError('Maximum slowdown must be in [0,1]')
    if not 0<=float(allowed_increase)<=1:
        raise ValueError('NLL tolerance must be between zero and one')
    characters=int(characters)
    if not 50<=characters<=3000:
        raise ValueError('Evaluate between 50 and 3000 characters per partition')
    model_dir=Path(model_dir) if model_dir else root/'models'/'dense-control'
    model=CharacterModel(model_dir)
    text=(root/'data'/'corpus.txt').read_text(encoding='utf-8')
    cutoff=int(len(text)*.88)
    holdout=text[cutoff:]
    midpoint=len(holdout)//2
    validation=holdout[max(0,midpoint-characters):midpoint]
    test=holdout[midpoint:midpoint+characters]
    if len(validation)<characters or len(test)<characters:
        raise ValueError('Corpus too short for sealed validation/test partitions')
    base_validation=model.score_text(validation)
    baseline_loss=base_validation['cross_entropy']
    reference=dict(model.operators)
    trial=dict(reference)
    rows=[]
    start=time.perf_counter()
    for name in ('input','recurrent','head'):
        attempts=[]
        for candidate in _candidate_ops(name,reference[name]):
            proposed={**trial,name:candidate}
            loss=model.score_text(validation,operators=proposed)['cross_entropy']
            record={'kind':candidate.kind,'rank':candidate.spec()['rank'],
                    'bytes':candidate.payload_bytes(),'validation_loss':loss,
                    'qualified':loss<=baseline_loss+allowed_increase}
            attempts.append((record,candidate))
        accepted=[(r,c) for r,c in attempts if r['qualified']]
        # Dense source candidate may fail if other layers already consumed the budget.
        if not accepted:raise AssertionError('Original dense representation no longer qualifies')
        picked,candidate=min(accepted,key=lambda a:(a[0]['bytes'],a[0]['validation_loss'],a[0]['kind']))
        trial[name]=candidate
        rows.append({'operator':name,'selected':picked,'alternatives':[r for r,_ in attempts]})
    # Time the complete neural execution, not isolated operators. Timing uses
    # validation text ONLY; the sealed test never participates in selection.
    # Alternate execution order to reduce systematic warm-cache/order bias.
    # One warmup before timing helps reduce cold Python/BLAS overhead.
    model.step(model.token_ids(validation)[0], np.zeros(model.state_dim,dtype=np.float32), reference)
    model.step(model.token_ids(validation)[0], np.zeros(model.state_dim,dtype=np.float32), trial)
    baseline_times=[]; candidate_times=[]
    token_ids=model.token_ids(validation[:128])
    for iteration in range(10):
        pairs=[('baseline',reference),('candidate',trial)]
        if iteration%2:pairs.reverse()
        for label,ops in pairs:
            state=np.zeros(model.state_dim,dtype=np.float32)
            t_start=time.perf_counter_ns()
            for token in token_ids:
                state,_,_=model.step(token,state,ops)
            ms=(time.perf_counter_ns()-t_start)/(1e6*len(token_ids))
            (baseline_times if label=='baseline' else candidate_times).append(ms)
    baseline_ms=float(np.median(baseline_times))
    candidate_ms=float(np.median(candidate_times))
    ratio=candidate_ms/max(baseline_ms,1e-12)
    passes=ratio<=1+float(max_slowdown)
    pre_guard_kinds={name:op.kind for name,op in trial.items()}
    if objective=='latency_guarded' and not passes:
        # Crucially, this is not a new operator path: reuse exactly the original
        # validated checkpoint. Keep the rejected candidate's evidence visible.
        trial=dict(reference)
        for row in rows:
            row['pre_guard_selected']=row['selected']
            row['selected']=next(r for r in row['alternatives'] if r['kind']=='dense')
    final_validation=model.score_text(validation,operators=trial)
    # Sealed test is touched only AFTER the candidate and optional latency gate.
    base_test=model.score_text(test)
    final_test=model.score_text(test,operators=trial)
    permutation=np.random.default_rng(20261019).permutation(len(test))
    shuffled=''.join(test[int(i)] for i in permutation)
    base_shuffled=model.score_text(shuffled)
    selected_shuffled=model.score_text(shuffled,operators=trial)
    timing={'method':'median ms per character, 10 alternating-order passes, validation text only, CPU/Python overhead included',
            'baseline_ms_per_character':baseline_ms,
            'candidate_ms_per_character':candidate_ms,
            'selected_ms_per_character':candidate_ms if (objective=='storage' or passes) else baseline_ms,
            'candidate_to_baseline_ratio':ratio,'platform':platform.platform()}
    decision={'objective':objective,'max_slowdown':float(max_slowdown),
              'candidate_within_latency_budget':bool(passes),
              'accepted':bool(objective=='storage' or passes),
              'reason':('storage_priority' if objective=='storage' else
                        'within_validation_latency_budget' if passes else 'reverted_to_dense_due_to_validation_latency'),
              'candidate_operator_kinds':pre_guard_kinds,
              'note':'Cost check is a noisy local CPU measurement, not a cross-hardware guarantee. Sealed test is never used for selection.'}
    original_bytes=model.status()['parameter_bytes']
    chosen_bytes=sum(op.payload_bytes() for op in trial.values())+model.embedding.nbytes+model.bias.nbytes+model.head_bias.nbytes
    report={'schema':1,'kind':'task_loss_operator_probe','source_checkpoint':model.manifest['checkpoint_id'],
      'corpus_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
      'selection_rule':'Validation-constrained minimum payload followed by optional validation-only end-to-end latency gate; sealed test excluded',
      'decision':decision,
      'allowed_increase':float(allowed_increase),
      'partitions':{'train_characters':cutoff,'validation_characters':len(validation),'test_characters':len(test),
                    'note':'Templated authored corpus repeats patterns; validation/test separation does not guarantee semantic independence'},
      'base_validation':base_validation,'base_test':base_test,
      'selected_validation':final_validation,'selected_test':final_test,
      'shuffled_negative_control':{'base':base_shuffled,'selected':selected_shuffled,
         'note':'Shuffled held-out characters are a distribution-shift stress case, not meaningful natural language.'},
      'original_payload_bytes':original_bytes,'selected_payload_bytes':chosen_bytes,
      'relative_payload_saving':1-chosen_bytes/original_bytes,
      'operators':rows,'timing':timing,'seconds':round(time.perf_counter()-start,3),
      'limitation':'Post-training candidate search on a tiny RNN; not open-ended equation discovery, large-model compression or a GPU performance result.'}
    if write_candidate:
        if objective=='latency_guarded' and not decision['accepted']:
            raise ValueError('No improved candidate passed the latency gate; refusing to export an unchanged model')
        artifact_id=hashlib.sha256(json.dumps({'checkpoint':report['source_checkpoint'],'operators':[r['selected'] for r in rows],
            'allowed_increase':float(allowed_increase),'characters':characters,'objective':objective,'max_slowdown':float(max_slowdown)},sort_keys=True).encode()).hexdigest()[:16]
        dest=root/'models'/'candidates'/('probe_'+artifact_id)
        if dest.exists():
            raise ValueError('Candidate already exists; preserving immutable checkpoint')
        dest.mkdir(parents=True)
        arrays={'embedding':model.embedding,'bias':model.bias,'head_bias':model.head_bias}
        specs={}
        for name,op in trial.items():
            specs[name]={'kind':op.kind,'shape':list(op.shape),'array_keys':list(op.arrays)}
            for key,value in op.arrays.items():arrays[f'{name}.{key}']=value
        np.savez_compressed(dest/'weights.npz',**arrays)
        manifest={**model.manifest,'checkpoint_id':artifact_id,'operators':specs,
                  'parent_checkpoint_id':report['source_checkpoint'],
                  'post_training_probe':{'allowed_increase':float(allowed_increase),'validation_loss':final_validation['cross_entropy'],
                    'sealed_test_loss':final_test['cross_entropy']}}
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        (dest/'probe.json').write_text(json.dumps(report,indent=2))
        # Numerical artifact readiness check; no auto-promotion.
        reloaded=CharacterModel(dest)
        check=reloaded.score_text(test)
        if abs(check['cross_entropy']-final_test['cross_entropy'])>1e-5:
            raise AssertionError('Candidate checkpoint changed inference result during export')
        report['candidate_path']=str(dest)
    return report
