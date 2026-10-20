"""PyTorch native training; export one NumPy operator artifact for all readers.
The trained model is deliberately tiny and is NOT a general chat system.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np

def train_model(corpus_path,outdir,kind='factorized',hidden=48,embedding_dim=24,rank=16,steps=220,seed=7,sequence_length=48,batch_size=20,layers=1):
    import torch
    from torch import nn
    layers=int(layers)
    if not 1<=layers<=4:raise ValueError('Train between 1 and 4 recurrent layers')
    torch.set_num_threads(min(2,torch.get_num_threads()))
    torch.manual_seed(int(seed));np.random.seed(int(seed))
    corpus=Path(corpus_path).read_text(encoding='utf-8')
    if len(corpus)<4000:raise ValueError('Training corpus requires 4000+ characters')
    vocab=sorted(set(corpus+' '));index={c:i for i,c in enumerate(vocab)}
    data=torch.tensor([index[c] for c in corpus],dtype=torch.long)
    cutoff=int(len(data)*.88);tr=data[:cutoff];vl=data[cutoff:]
    if len(vl)<sequence_length+2:raise ValueError('Not enough validation characters')
    class Layer(nn.Module):
        def __init__(self,inputs,outputs):
            super().__init__();self.kind=kind
            if kind=='dense':self.w=nn.Parameter(torch.randn(outputs,inputs)*.07)
            elif kind=='factorized':
                r=min(int(rank),inputs,outputs)
                # Match dense initialization's effective std (.07) in A@B.
                # Var[(AB)ij] = r * Var(A) * Var(B), so s = sqrt(.07)/r**.25.
                sigma=(.07**.5)/(r**.25)
                self.a=nn.Parameter(torch.randn(outputs,r)*sigma)
                self.b=nn.Parameter(torch.randn(r,inputs)*sigma)
            else:raise ValueError('training supports dense or factorized')
        def forward(self,x):
            return (x @ self.w.T) if self.kind=='dense' else ((x @ self.b.T) @ self.a.T)
        def export(self,prefix,arrays):
            if self.kind=='dense':
                arrays[prefix+'.w']=self.w.detach().cpu().numpy().astype('float32');keys=['w']
            else:
                for k in ('a','b'):arrays[prefix+'.'+k]=getattr(self,k).detach().cpu().numpy().astype('float32')
                keys=['a','b']
            return {'kind':self.kind,'shape':[self.out_dim,self.in_dim],'array_keys':keys}
    # Attach dimensions for a direct serialization contract.
    def mk(i,o):
        a=Layer(i,o);a.in_dim=i;a.out_dim=o;return a
    class TinyRNN(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(len(vocab),embedding_dim)
            self.input=mk(embedding_dim,hidden);self.recurrent=mk(hidden,hidden);self.head=mk(hidden,len(vocab))
            self.deep=nn.ModuleList([nn.ModuleDict({'input':mk(hidden,hidden),'recurrent':mk(hidden,hidden)}) for _ in range(layers-1)])
            self.bias=nn.Parameter(torch.zeros(hidden));self.head_bias=nn.Parameter(torch.zeros(len(vocab)))
        def forward(self,x):
            states=[torch.zeros(x.size(0),hidden) for _ in range(layers)]
            outs=[]
            for t in range(x.shape[1]):
                layer_input=self.embedding(x[:,t])
                for k in range(layers):
                    if k==0:
                        new=torch.tanh(self.input(layer_input)+self.recurrent(states[k])+self.bias)
                    else:
                        new=torch.tanh(self.deep[k-1]['input'](layer_input)+self.deep[k-1]['recurrent'](states[k]))
                    states[k]=new
                    layer_input=new
                outs.append(self.head(layer_input)+self.head_bias)
            return torch.stack(outs,dim=1)
    model=TinyRNN();optim=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=.0001)
    def batch(source,n):
        indexes=torch.randint(0,len(source)-sequence_length-1,(n,))
        offsets=torch.arange(sequence_length)
        x=source[indexes[:,None]+offsets];y=source[indexes[:,None]+offsets+1]
        return x,y
    history=[];start=time.perf_counter()
    for step in range(1,int(steps)+1):
        model.train();x,y=batch(tr,batch_size)
        loss=torch.nn.functional.cross_entropy(model(x).reshape(-1,len(vocab)),y.reshape(-1))
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        if step==1 or step%max(1,int(steps)//10)==0 or step==int(steps):history.append({'step':step,'train_loss':float(loss.detach())})
    model.eval()
    with torch.no_grad():
        torch.manual_seed(int(seed)+441);x,y=batch(vl,40)
        validation=float(torch.nn.functional.cross_entropy(model(x).reshape(-1,len(vocab)),y.reshape(-1)))
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    arrays={'embedding':model.embedding.weight.detach().numpy().astype('float32'),
            'bias':model.bias.detach().numpy().astype('float32'),'head_bias':model.head_bias.detach().numpy().astype('float32')}
    ops={name:getattr(model,name).export(name,arrays) for name in ('input','recurrent','head')}
    for layer,block in enumerate(model.deep,start=1):
        for name in ('input','recurrent'):
            key=f'{name}_{layer}'
            ops[key]=block[name].export(key,arrays)
    checkpoint_id=hashlib.sha256((corpus+str(seed)+kind+str(steps)+str(hidden)+str(rank)+str(layers)+'variance_matched_v1').encode()).hexdigest()[:16]
    manifest={'schema':1,'checkpoint_id':checkpoint_id,'kind':'stacked-rnn' if layers>1 else 'simple-rnn','layers':layers,'vocab':vocab,'hidden':hidden,'embedding_dim':embedding_dim,'operators':ops,
        'training':{'corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest(),'train_chars':len(tr),'validation_chars':len(vl),
            'steps':int(steps),'seed':int(seed),'layers':layers,'initializer':'variance_matched_v1','rank':int(rank) if kind=='factorized' else None,
            'batch_size':batch_size,'sequence_length':sequence_length,'train_loss':history[-1]['train_loss'],
            'validation_loss':validation,'seconds':round(time.perf_counter()-start,3), 'corpus_note':'Small authored demonstration corpus; quality claims limited to this domain.'}}
    np.savez_compressed(outdir/'weights.npz',**arrays)
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (outdir/'training_history.json').write_text(json.dumps(history,indent=2))
    # Independent inference verification: NumPy operator applies identical recurrence.
    from .model import CharacterModel
    exported=CharacterModel(outdir)
    seq='The atlas learns.'
    state=np.zeros(hidden*layers,dtype=np.float32)
    for i in exported.token_ids(seq):state,logits,_=exported.step(i,state)
    with torch.no_grad():
        ids=torch.tensor([exported.token_ids(seq)])
        torch_out=model(ids)[0,-1].numpy()
    err=float(np.max(np.abs(logits-torch_out)))
    if err>1e-4:raise AssertionError(f'training/inference parity failure: {err}')
    manifest['training']['numpy_torch_max_error']=err
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    return {'path':str(outdir),'training':manifest['training'],'checkpoint_id':checkpoint_id,'parity_max_error':err,
        'parameter_bytes':exported.status()['parameter_bytes']}
