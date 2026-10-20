"""Authoritative graph metadata used by both GUI and API. Not a second execution graph."""
NODES=[
 {'id':'input','name':'Input Text','type':'io','x':35,'y':280,'about':'Incoming UTF-8 text is encoded into character IDs using the checkpoint vocabulary.'},
 {'id':'embedding','name':'Token Embedding','type':'neural','x':270,'y':280,'about':'One learned vector per vocabulary character. Weights come from the model checkpoint.'},
 {'id':'input_op','name':'Input Operator','type':'operator','operator':'input','x':525,'y':160,'about':'Maps a token embedding into neural state space. Dense or factorized according to the checkpoint.'},
 {'id':'state','name':'Recurrent State','type':'state','x':520,'y':415,'about':'A persistent hidden vector; state updates once per decoded character.'},
 {'id':'recurrent_op','name':'State Operator','type':'operator','operator':'recurrent','x':755,'y':420,'about':'Transforms the previous hidden state into a contribution to the new state.'},
 {'id':'activation','name':'Activation / tanh','type':'neural','x':1005,'y':275,'about':'Combines current input and previous state, adds bias and applies tanh.'},
 {'id':'head','name':'Output Projection','type':'operator','operator':'head','x':1255,'y':280,'about':'Turns hidden state into a score for every vocabulary character.'},
 {'id':'sampler','name':'Token Sampler','type':'io','x':1510,'y':280,'about':'Softmax probabilities followed by seeded sampling. Each chosen token feeds the next step.'},
 {'id':'documents','name':'Evidence Store','type':'memory','x':290,'y':650,'about':'Immutable text documents with content hashes and lexical retrieval. This does not modify model weights.'},
 {'id':'equation','name':'Equation Discovery','type':'research','x':780,'y':650,'about':'Fits and persists provenance-linked equations from numerical CSV evidence under a declared basis with held-out testing. No neural weight updates.'},
 {'id':'bench','name':'Evaluation / Bench','type':'research','x':1250,'y':650,'about':'Compares mathematical representation errors, task-level loss, storage bytes and real host-local latency.'},
 {'id':'file_import','name':'Document Import','type':'memory','x':35,'y':650,'about':'Accepts local document files through bounded parsers, then passes extracted text to the one SQLite evidence store. No neural weight updates.'},
 {'id':'answer_router','name':'Answer Routing','type':'research','x':530,'y':890,'about':'Typed resolver: use an exact qualified equation, source evidence, or an explicit abstention. Experimental generation is never disguised as source-backed knowledge.'},
 {'id':'answer_output','name':'Answer Result','type':'io','x':1260,'y':890,'about':'Typed output with route, source IDs, equation validation errors, and disclosure. Grounded responses preserve original evidence.'},
 {'id':'composition','name':'Equation Composition','type':'research','x':1010,'y':900,'about':'Evaluates two to six already-qualified source-bound equations in order. Aborts when a stage leaves its observed domain. Does not establish unit compatibility or combined uncertainty.'},
 {'id':'diagnostics','name':'Experiment & Event Log','type':'research','x':1510,'y':650,'about':'Sanitized JSONL action and latency logs plus immutable experiment reports. Text and documents are not logged.'}
]
EDGES=[('input','embedding'),('embedding','input_op'),('input_op','activation'),('state','recurrent_op'),('recurrent_op','activation'),
       ('activation','state'),('activation','head'),('head','sampler'),('sampler','input'),
       ('file_import','documents'),('documents','equation'),('equation','bench'),('bench','diagnostics'),('input','answer_router'),('answer_router','documents'),('answer_router','equation'),('documents','answer_output'),('equation','answer_output'),('answer_router','answer_output'),('equation','composition'),('composition','answer_output')]

def graph(model=None):
    """Derive visual topology from the checkpoint's actual recurrent depth.

    Coordinates are presentation only. For legacy checkpoints the original
    graph and IDs remain unchanged.
    """
    layers=int(getattr(model,'layers',1))
    nodes=[dict(n) for n in NODES]
    edges=list(EDGES)
    if layers>1:
        extra=layers-1
        offset=600*extra
        for node in nodes:
            if node['id'] in ('head','sampler','bench','diagnostics'):
                node['x']+=offset
        edges.remove(('activation','head'))
        previous='activation'
        for k in range(1,layers):
            x=1230+600*(k-1)
            prefix=f'layer{k}'
            nodes.extend([
                {'id':prefix+'_input_op','name':f'Layer {k+1} Input','type':'operator','operator':f'input_{k}',
                 'trace_key':f'input_{k}','x':x,'y':160,'about':f'Layer {k+1} maps the prior layer activation through its canonical input operator.'},
                {'id':prefix+'_state','name':f'Layer {k+1} State','type':'state',
                 'trace_key':f'activation_{k}','x':x,'y':420,'about':f'Recurrent state of layer {k+1}, preserved across token steps.'},
                {'id':prefix+'_recurrent_op','name':f'Layer {k+1} Recurrence','type':'operator',
                 'operator':f'recurrent_{k}','trace_key':f'recurrent_{k}','x':x+220,'y':420,'about':f'Recurrent transformation of layer {k+1}.'},
                {'id':prefix+'_activation','name':f'Layer {k+1} Activation','type':'neural',
                 'trace_key':f'activation_{k}','x':x+425,'y':275,'about':f'Tanh combination for layer {k+1}; receives prior layer and previous local state.'}
            ])
            edges.extend([(previous,prefix+'_input_op'),(prefix+'_input_op',prefix+'_activation'),
                          (prefix+'_state',prefix+'_recurrent_op'),(prefix+'_recurrent_op',prefix+'_activation'),
                          (prefix+'_activation',prefix+'_state')])
            previous=prefix+'_activation'
        edges.append((previous,'head'))
    return {'schema':1,'nodes':nodes,'edges':[{'from':a,'to':b} for a,b in edges],
            'layers':layers,'description':'Checkpoint-derived mathematical graph inspector. Coordinates do not change execution.'}
