"""Typed request resolution across existing model and evidence authorities.

This is not a second neural model or evidence store. Grounded mode produces
extractive replies with exact source provenance and abstains when unsupported.
Explicit experimental mode calls the existing character model, labelled as such.
"""
import re

_EQUATION_REQUEST = re.compile(
    r'^\s*(?:predict|evaluate|calculate)\s+equation\s*#?([1-9][0-9]*)\s+(?:at\s+)?(?:x\s*=\s*)?([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*[?.]?\s*$',
    re.I)

_CHAIN_REQUEST=re.compile(r'^\s*(?:chain|compose)\s+equations?\s+([0-9][0-9, ]{0,40})\s+at\s+([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*[?.]?\s*$', re.I)

def respond(service, question, mode='grounded', max_new=65, temperature=.65, seed=12):
    """Return one typed result and a transparent execution trace.

    These are mutually exclusive, explicitly disclosed capabilities.
    Never fabricate a source-based answer from neural continuation.
    """
    text=str(question).strip()
    if not text or len(text)>500:
        raise ValueError('Enter 1–500 characters')
    if mode not in ('grounded','experimental'):
        raise ValueError('Mode must be grounded or experimental')
    if mode=='experimental':
        if service.model is None:raise ValueError('Experimental model unavailable')
        out=service.model.generate(text,max_new,temperature,seed)
        return {'schema':'atlas-answer-v1','mode':mode,'route':'character_neural',
                'response':out['response'],'supported':False,
                'evidence':[],'equation':None,
                'trace':['input','embedding','input_op','recurrent_op','activation','head','sampler'],
                'disclosure':'Ungrounded sampled character continuation. It may produce gibberish and must not be treated as a factual answer.',
                'neural':out}
    chain=_CHAIN_REQUEST.fullmatch(text)
    if chain:
        ids=[int(t) for t in re.findall(r'\d+',chain.group(1))]
        result=service.db.compose_equations(ids,chain.group(2))
        if result['supported']:
            answer=(f"Composed {len(ids)} equations from x={result['initial_x']:g}: "
                    f"estimated output {result['value']:.6g}. "
                    "Each source and intermediate value is shown in the execution trace; units and accumulated error are unverified.")
        else:answer='Composition refused: '+result['reason']
        return {'schema':'atlas-answer-v1','mode':mode,'route':'composed_equations',
                'response':answer,'supported':result['supported'],'evidence':[],
                'equation':result,'trace':['input','answer_router','equation','composition','answer_output'],
                'disclosure':result['disclosure']}
    equation=_EQUATION_REQUEST.fullmatch(text)
    if equation:
        eid=int(equation.group(1));x=float(equation.group(2))
        result=service.db.predict_equation(eid,x)
        if result['supported']:
            answer=(f"Equation #{eid} predicts {result['value']:.6g} at x={x:g}. "
                    f"Held-out RMSE: {result['validation_rmse']:.6g}. "
                    f"Source #{result['source_id']}: {result['title']}.")
        else:
            answer=f"I cannot evaluate equation #{eid} at x={x:g}: {result['reason']}"
        return {'schema':'atlas-answer-v1','mode':mode,'route':'qualified_equation',
                'response':answer,'supported':bool(result['supported']),
                'evidence':[],'equation':result,
                'trace':['input','equation','answer_output'],
                'disclosure':'A bounded fitted numerical equation, not a verified physical law.'}
    found=service.db.answer_sentences(text,3)
    if not found:
        return {'schema':'atlas-answer-v1','mode':mode,'route':'abstain',
                'response':'I do not have a matching source for that question. Import relevant documents, or inspect experimental character generation separately.',
                'supported':False,'evidence':[],'equation':None,
                'trace':['input','answer_router','answer_output'],
                'disclosure':'No source match. The system refuses to substitute ungrounded neural text.'}
    first=found[0]
    # Return evidence explicitly. Never silently adjudicate conflicting documents.
    multiple=len({p['source_id'] for p in found})>1
    lead=('Multiple sources matched; their claims have NOT been reconciled. '
          if multiple else 'A matching source sentence was found. ')
    return {'schema':'atlas-answer-v1','mode':mode,'route':'source_excerpt',
            'response':lead+f"From “{first['title']}” (source #{first['source_id']}): {first['text']}",
            'supported':True,'evidence':found,'equation':None,
            'trace':['input','answer_router','documents','answer_output'],
            'disclosure':'Quoted source sentence selected by conservative word overlap. Potentially conflicting sources are NOT reconciled. This is not a verified answer or neural learning.'}
