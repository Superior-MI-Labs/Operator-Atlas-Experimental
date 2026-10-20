"""One mathematical operator contract for training artifacts and inference.

All operators accept input (..., input_dimension) and emit (..., output_dimension).
No execution path constructs A @ B for a factorized operator.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np

KINDS = ('dense', 'factorized', 'int8', 'factorized_sparse')

@dataclass(frozen=True)
class Operator:
    name: str
    kind: str
    shape: tuple[int, int]  # output, input
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError('unsupported operator: '+self.kind)
        o,i = self.shape
        if min(o,i) < 1: raise ValueError('invalid shape')
        expected={'dense':{'w':(o,i)}, 'int8':{'q':(o,i),'scale':()},
                  'factorized':{'a':(o,None),'b':(None,i)},
                  'factorized_sparse':{'a':(o,None),'b':(None,i),'rows':(None,),'cols':(None,),'values':(None,)}}[self.kind]
        for key,sh in expected.items():
            if key not in self.arrays: raise ValueError(f'{self.name}: missing {key}')
            ar=np.asarray(self.arrays[key]);
            if ar.ndim!=len(sh) or any(a!=b for a,b in zip(ar.shape,sh) if b is not None):
                raise ValueError(f'{self.name}: {key} dimension mismatch')
        if self.kind.startswith('factorized') and self.arrays['a'].shape[1]!=self.arrays['b'].shape[0]:
            raise ValueError('factor rank mismatch')
        if self.kind=='factorized_sparse':
            r,c,v=(np.asarray(self.arrays[k]) for k in ('rows','cols','values'))
            if not(len(r)==len(c)==len(v)) or np.any(r<0) or np.any(r>=o) or np.any(c<0) or np.any(c>=i):
                raise ValueError('invalid sparse coordinates')

    def apply(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape[-1]!=self.shape[1]: raise ValueError('input dimension mismatch')
        if self.kind=='dense': return x @ self.arrays['w'].T
        if self.kind=='int8': return (x @ self.arrays['q'].astype(np.float32).T)*float(self.arrays['scale'])
        y=(x @ self.arrays['b'].T) @ self.arrays['a'].T
        if self.kind=='factorized_sparse':
            flat=x.reshape(-1,x.shape[-1]); y2=y.reshape(-1,self.shape[0]).copy()
            r,c,v=(self.arrays[k] for k in ('rows','cols','values'))
            for row,col,val in zip(r,c,v): y2[:,int(row)] += flat[:,int(col)]*float(val)
            y=y2.reshape(x.shape[:-1]+(self.shape[0],))
        return y

    def payload_bytes(self):
        return sum(np.asarray(a).nbytes for a in self.arrays.values())

    def dense_equivalent(self):
        return self.apply(np.eye(self.shape[1],dtype=np.float32)).T

    def spec(self):
        return {'name':self.name,'kind':self.kind,'shape':list(self.shape),'payload_bytes':self.payload_bytes(),
                'rank':int(self.arrays['a'].shape[1]) if self.kind.startswith('factorized') else None,
                'nonzeros':len(self.arrays['values']) if self.kind=='factorized_sparse' else 0}

    def heatmap(self,max_side=12):
        matrix=self.dense_equivalent()
        idx0=np.linspace(0,matrix.shape[0]-1,min(max_side,matrix.shape[0])).astype(int)
        idx1=np.linspace(0,matrix.shape[1]-1,min(max_side,matrix.shape[1])).astype(int)
        return np.round(matrix[np.ix_(idx0,idx1)],4).tolist()

def from_artifact(name,spec,archive):
    arrays={key:np.array(archive[f'{name}.{key}']) for key in spec['array_keys']}
    return Operator(name,spec['kind'],tuple(spec['shape']),arrays)

def factorize(name,w,rank,sparse_budget=0):
    w=np.asarray(w,dtype=np.float32)
    rank=max(1,min(int(rank),min(w.shape)))
    def truncated(target):
        u,s,vh=np.linalg.svd(target,full_matrices=False)
        return (u[:,:rank]*s[:rank]).astype(np.float32), vh[:rank,:].astype(np.float32)
    if not sparse_budget:
        a,b=truncated(w)
        return Operator(name,'factorized',tuple(w.shape),{'a':a,'b':b})
    # Alternating low-rank/sparse fit avoids asking truncated SVD to absorb large
    # localized outliers. Heuristic only; no recovery guarantee.
    k=min(max(1,int(sparse_budget)),w.size)
    sparse=np.zeros_like(w)
    for _ in range(24):
        a,b=truncated(w-sparse)
        residual=w-a@b
        idx=np.argpartition(np.abs(residual).ravel(),-k)[-k:]
        sparse=np.zeros(w.size,dtype=np.float32)
        sparse[idx]=residual.ravel()[idx]
        sparse=sparse.reshape(w.shape)
    a,b=truncated(w-sparse)
    residual=w-a@b
    idx=np.argpartition(np.abs(residual).ravel(),-k)[-k:]
    row,col=np.unravel_index(idx,w.shape)
    return Operator(name,'factorized_sparse',tuple(w.shape),{'a':a,'b':b,
        'rows':row.astype(np.int32),'cols':col.astype(np.int32),'values':residual[row,col].astype(np.float32)})

def quantize(name,w):
    w=np.asarray(w,dtype=np.float32); scale=np.float32(np.max(np.abs(w))/127.0 or 1.0)
    return Operator(name,'int8',tuple(w.shape),{'q':np.clip(np.rint(w/scale),-127,127).astype(np.int8),'scale':np.array(scale)})
