"""Validation-constrained operator choice. No optimizer can guarantee arbitrary OOD performance."""
import numpy as np
from .operators import Operator,factorize,quantize

def select_operator(weights,validation_inputs,max_relative_error=.025,shifted_inputs=None):
    w=np.asarray(weights,dtype=np.float32);x=np.asarray(validation_inputs,dtype=np.float32)
    if w.ndim!=2 or x.ndim!=2 or x.shape[1]!=w.shape[1]:raise ValueError('matrix and input shape mismatch')
    if not (0<=max_relative_error<=1):raise ValueError('relative-error budget outside [0,1]')
    opts=[Operator('candidate','dense',w.shape,{'w':w}),quantize('candidate',w)]
    for rank in sorted(set([2,4,8,16,min(w.shape)])):
        if rank>min(w.shape):continue
        opts.append(factorize('candidate',w,rank))
        if rank<min(w.shape):opts.append(factorize('candidate',w,rank,max(2,int(w.size*.01))))
    target=x@w.T;norm=max(float(np.linalg.norm(target)),1e-12)
    results=[]
    for op in opts:
        error=float(np.linalg.norm(op.apply(x)-target)/norm)
        shifted=None
        if shifted_inputs is not None:
            sx=np.asarray(shifted_inputs,dtype=np.float32);yt=sx@w.T
            shifted=float(np.linalg.norm(op.apply(sx)-yt)/max(float(np.linalg.norm(yt)),1e-12))
        results.append({'operator':op,'label':op.kind+(f' rank {op.spec()["rank"]}' if op.spec()['rank'] else ''),
                        'validation_error':error,'shifted_error':shifted,'bytes':op.payload_bytes(),'qualified':error<=max_relative_error})
    # Min bytes, then error, then deterministic label; never use shifted holdout to make selection.
    eligible=[r for r in results if r['qualified']]
    chosen=min(eligible,key=lambda r:(r['bytes'],r['validation_error'],r['label']))
    return chosen['operator'],{'selected':chosen['label'],'selected_bytes':chosen['bytes'],'max_relative_error':max_relative_error,
      'alternatives':[{'kind':r['label'],'bytes':r['bytes'],'validation_error':r['validation_error'],
                       'shifted_error':r['shifted_error'],'qualified':r['qualified']} for r in results]}
