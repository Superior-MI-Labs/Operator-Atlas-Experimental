import math
"""Small character-level autoregressive recurrent decoder; NumPy inference.

Traces expose actual state, operator outputs and logits, not visual approximations.
"""
import json
from pathlib import Path
import numpy as np
from .operators import from_artifact

class CharacterModel:
    def __init__(self, model_dir):
        path=Path(model_dir)
        manifest=json.loads((path/'manifest.json').read_text())
        self.manifest=manifest
        self.vocab=manifest['vocab']; self.char_to_id={c:i for i,c in enumerate(self.vocab)}
        with np.load(path/'weights.npz',allow_pickle=False) as ar:
            self.embedding=np.array(ar['embedding'],dtype=np.float32)
            self.bias=np.array(ar['bias'],dtype=np.float32)
            self.head_bias=np.array(ar['head_bias'],dtype=np.float32)
            self.operators={key:from_artifact(key,sp,ar) for key,sp in manifest['operators'].items()}
        self.hidden=int(manifest['hidden']);self.layers=int(manifest.get('layers',1));self.state_dim=self.hidden*self.layers
        if not 1<=self.layers<=8:raise ValueError('Unsupported recurrent depth')
        for layer in range(1,self.layers):
            for name in (f'input_{layer}',f'recurrent_{layer}'):
                if name not in self.operators:raise ValueError('Incomplete stacked checkpoint: '+name)
        self.unknown=self.char_to_id.get(' ',0)
    def token_ids(self,text): return [self.char_to_id.get(c,self.unknown) for c in text]
    def step(self,idx,state,operators=None):
        """One authoritative recurrent step, including optional deeper layers.

        State shape is (layers * hidden,). Layer zero retains the v0.3
        checkpoint semantics and trace field names.
        """
        operators = self.operators if operators is None else operators
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=(self.state_dim,):raise ValueError('Invalid recurrent state shape')
        previous=state.reshape(self.layers,self.hidden)
        embed=self.embedding[idx]
        vectors={'embedding':embed}
        new=[];input_vec=embed
        for layer in range(self.layers):
            suffix='' if layer==0 else f'_{layer}'
            a=operators['input'+suffix].apply(input_vec)
            b=operators['recurrent'+suffix].apply(previous[layer])
            next_state=np.tanh(a+b+self.bias if layer==0 else a+b).astype(np.float32)
            vectors['input'+suffix]=a
            vectors['recurrent'+suffix]=b
            vectors['activation'+suffix]=next_state
            new.append(next_state)
            input_vec=next_state
        logits=operators['head'].apply(input_vec)+self.head_bias
        vectors['logits']=logits
        return np.concatenate(new),logits,vectors
    def generate(self,prompt,max_new=64,temperature=0.8,seed=12):
        if not prompt: prompt=' '
        max_new=int(max(1,min(160,max_new)))
        temperature=float(temperature)
        if not math.isfinite(temperature) or not 0.05<=temperature<=2.0:
            raise ValueError('Temperature must be finite and between 0.05 and 2.0')
        rng=np.random.default_rng(int(seed));state=np.zeros(self.state_dim,dtype=np.float32)
        trace=[];logits=None
        for cidx in self.token_ids(prompt): state,logits,t=self.step(cidx,state)
        out=[]
        for n in range(max_new):
            scaled=logits/temperature; probs=np.exp(scaled-np.max(scaled));probs/=probs.sum()
            idx=int(rng.choice(len(probs),p=probs))
            out.append(self.vocab[idx]);state,logits,t=self.step(idx,state)
            if n<10 or n==max_new-1:
                top=np.argsort(-probs)[:4]
                trace.append({'step':n+1,'token':self.vocab[idx], 'top':[{'token':self.vocab[int(k)],'probability':round(float(probs[k]),4)} for k in top],
                              'activation_strength':round(float(np.linalg.norm(t['activation'])),4)})
        return {'response':''.join(out),'prompt':prompt,'generated_characters':len(out),
                'trace':trace, 'last_vectors':{k:np.asarray(v[:32]).round(4).tolist() for k,v in t.items()},
                'state_size_bytes':int(state.nbytes),'note':'Character-level sampled continuation from a small trained recurrent neural network. Not a factual question-answering system.'}
    def score_text(self, text, operators=None, limit=1200):
        """Teacher-forced next-character NLL through exactly the inference recurrence.

        This does not measure general language understanding. The first character
        initializes state; later characters are scored against the preceding state.
        """
        ids=self.token_ids(str(text)[:max(2,min(int(limit),10000))])
        if len(ids)<2:raise ValueError('Need at least two characters to score')
        state=np.zeros(self.state_dim,dtype=np.float32)
        total=0.0
        for current,expected in zip(ids[:-1],ids[1:]):
            state,logits,_=self.step(current,state,operators=operators)
            z=np.asarray(logits,dtype=np.float64)
            maxval=float(z.max())
            log_denom=maxval+np.log(np.exp(z-maxval).sum())
            total+=float(log_denom-z[expected])
        return {'cross_entropy':total/(len(ids)-1),'characters':len(ids)-1,
                'unknown_fraction':sum(c not in self.char_to_id for c in str(text)[:len(ids)])/len(ids)}

    def status(self):
        ops={key:v.spec() for key,v in self.operators.items()}
        return {'checkpoint':self.manifest.get('checkpoint_id'), 'vocab_size':len(self.vocab), 'hidden':self.hidden,'layers':self.layers,'state_size_bytes':self.state_dim*4,
            'operator_specs':ops,'parameter_bytes':sum(x.payload_bytes() for x in self.operators.values())+self.embedding.nbytes+self.bias.nbytes+self.head_bias.nbytes,
            'training':self.manifest.get('training',{}),'kind':'character-level recurrent neural network'}
