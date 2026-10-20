"""Public synthetic fixture evaluation; uses real single service / evidence actions.

No personal source files, live databases, private prompts, or new inference owner.
"""
import hashlib
from pathlib import Path
from .api import AtlasService
from .importers import decode_document

FIXTURES=('research_notebook.md','unrelated_notes.md','calibration.csv','secondary_calibration.csv','random_control.csv')

def fixture_demonstration(root, isolated_root):
    root=Path(root);scratch=Path(isolated_root);scratch.mkdir(parents=True,exist_ok=True)
    # The scratch service reads the *released checkpoint* and owns a new temporary DB.
    (scratch/'models').mkdir(exist_ok=True)
    import shutil
    shutil.copytree(root/'models'/'starter',scratch/'models'/'starter')
    service=AtlasService(scratch)
    question='Where was the Cloudglass Observatory reference prism stored?'
    before=service.process('ask_sources',{'question':question})
    response_before=service.process('respond',{'question':question})
    assert not before['found']
    records=[]
    for name in FIXTURES:
        path=root/'examples'/'fixtures'/name
        raw=path.read_bytes()
        text,digest,mime=decode_document(name,raw)
        outcome=service.db.ingest(name,text,source_name=name,media_type=mime,original_sha256=digest)
        records.append({'name':name,'source_id':outcome['source_id'],'sha256':hashlib.sha256(raw).hexdigest()})
    after=service.process('ask_sources',{'question':question})
    response_after=service.process('respond',{'question':question})
    assert response_before['route']=='abstain' and response_after['route']=='source_excerpt'
    assert 'KAPPA-47' in response_after['response']
    assert after['found'] and any(p['source_id']==records[0]['source_id'] for p in after['passages'])
    numeric=service.process('learn_equation',{'source_id':records[2]['source_id'], 'x_column':'coordinate','y_column':'reading'})
    prediction=service.process('predict_equation',{'equation_id':numeric['equation_id'],'x':0.375})
    assert prediction['supported']
    response_math=service.process('respond',{'question':f"predict equation {numeric['equation_id']} at 0.375"})
    assert response_math['route']=='qualified_equation' and response_math['supported']
    outside=service.process('predict_equation',{'equation_id':numeric['equation_id'],'x':50})
    assert not outside['supported']
    bad_control_rejected=False
    try:service.process('learn_equation',{'source_id':records[4]['source_id'],'x_column':'coordinate','y_column':'reading'})
    except ValueError:bad_control_rejected=True
    assert bad_control_rejected
    secondary=service.process('learn_equation',{'source_id':records[3]['source_id'],
        'x_column':'measurement','y_column':'calibrated'})
    chained=service.process('compose_equations',{'equation_ids':[numeric['equation_id'],secondary['equation_id']],'x':.375})
    assert chained['supported'] and len(chained['steps'])==2
    response_chain=service.process('respond',{'question':f"chain equations {numeric['equation_id']} {secondary['equation_id']} at 0.375"})
    assert response_chain['route']=='composed_equations' and response_chain['supported']
    failed_chain=service.process('compose_equations',{'equation_ids':[numeric['equation_id'],secondary['equation_id']],'x':10})
    assert not failed_chain['supported'] and len(failed_chain['steps'])==1
    import math
    target=1.4-.5*.375**2+2*math.sin(2*.375)
    assert abs(prediction['value']-target)<.03
    return {'schema':'operator-atlas-fixture-study-v1','experiment_type':'synthetic_public_fixture_not_independent_language_benchmark',
            'fixture_hashes':records,'question':question,
            'before':{'found':before['found'],'method':before['method'],'response_route':response_before['route'],'supported':response_before['supported']},
            'after':{'found':after['found'],'source_ids':[p['source_id'] for p in after['passages']],
                     'matched_source':records[0]['source_id'],'response_route':response_after['route'],'supported':response_after['supported']},
            'composition':{'first_equation':numeric['equation_id'],'second_equation':secondary['equation_id'],
                           'supported':chained['supported'],'response_route':response_chain['route'],
                           'intermediate_value':chained['steps'][0]['value'],'final_value':chained['value'],
                           'outside_range_refused':not failed_chain['supported'],
                           'disclosure':chained['disclosure']},
            'equation':{'equation_id':numeric['equation_id'],'source_id':numeric['source_id'],
                        'formula':numeric['equation'],'terms':numeric['terms'],
                        'validation_rmse':numeric['validation_rmse'],
                        'mean_baseline_rmse':numeric['mean_baseline_rmse'],
                        'within_range_estimate':prediction['value'],
                        'independent_synthetic_truth':target,
                        'absolute_error':abs(prediction['value']-target),
                        'response_route':response_math['route'],
                        'outside_range_refused':not outside['supported'],
                        'random_control_refused':bad_control_rejected},
            'evidence':service.db.stats(),
            'limits':'Synthetic data; equation search uses declared 13-term grammar, and chained equations do not establish unit compatibility or cumulative error. No neural weights changed.'}


def evidence_quality_study(root, scratch):
    """Controlled, reproducible source-answer and refusal test; no model training.

    The handcrafted questions intentionally share source words. This evaluates
    citation plumbing and cautious lexical retrieval, NOT linguistic competence.
    """
    import json
    root=Path(root); scratch=Path(scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    cases=json.loads((root/'examples/fixtures/evidence_qa_cases.json').read_text())
    service=AtlasService(scratch)
    queries=cases['positive']+cases['negative']
    before=[service.process('respond',{'question':r['question']}) for r in queries]
    assert all(r['route']=='abstain' and not r['supported'] for r in before)
    sources=[]
    for name in ('research_notebook.md','unrelated_notes.md','evidence_qa.md'):
        file=root/'examples/fixtures'/name
        text,digest,mime=decode_document(name,file.read_bytes())
        stored=service.db.ingest(name,text,media_type=mime,original_sha256=digest)
        sources.append({'source_id':stored['source_id'],'sha256':stored['sha256'],'name':name})
    passed=[];missed=[];negative_correct=[];negative_wrong=[];baseline_false_positives=[]
    for item in cases['positive']:
        output=service.process('respond',{'question':item['question']})
        correct=(output['route']=='source_excerpt' and output['supported'] and
                 item['expected'].lower() in output['response'].lower() and output['evidence'] and
                 output['evidence'][0]['sha256'] in {s['sha256'] for s in sources})
        (passed if correct else missed).append({'question':item['question'],'route':output['route'],
                  'expected':item['expected'],'source_id':output['evidence'][0]['source_id'] if output['evidence'] else None})
    for item in cases['negative']:
        output=service.process('respond',{'question':item['question']})
        (negative_correct if not output['supported'] else negative_wrong).append({'question':item['question'], 'route':output['route']})
        if service.db.search(item['question']):baseline_false_positives.append(item['question'])
    # Conflicting source claims must remain separately cited and unadjudicated.
    conflicting=AtlasService(scratch/'conflict')
    conflicting.db.ingest('Account A','The observatory reference lens was stored in vault MAPLE-21 during inspection.')
    conflicting.db.ingest('Account B','The observatory reference lens was stored in vault BIRCH-92 during inspection.')
    contradiction=conflicting.process('respond',{'question':'Where was the observatory reference lens stored?'})
    assert contradiction['supported'] and len({s['source_id'] for s in contradiction['evidence']})==2
    assert 'NOT been reconciled' in contradiction['response']
    return {'schema':'operator-atlas-evidence-study-v1','data_type':'hand-authored fictional fixtures',
            'positive_total':len(cases['positive']),'positive_correct':len(passed),'positive_missed':missed,
            'negative_total':len(cases['negative']),'negative_abstained':len(negative_correct),
            'negative_errors':negative_wrong,'naive_chunk_search_false_positive_count':len(baseline_false_positives),
            'before_all_abstained':True,'sources':sources,'conflict_reported':True,
            'limitations':cases['limits']+' Quoting a source does not prove truth, and the test does not cover paraphrase, contradiction detection, or independent comprehension.'}
