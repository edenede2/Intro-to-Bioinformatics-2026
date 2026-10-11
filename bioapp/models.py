"""Tiny NumPy neural networks trained only on explicit synthetic teaching data."""
import numpy as np
from .sequences import PROTEIN


def softmax(x):
    x=x-x.max(axis=1,keepdims=True); p=np.exp(x); return p/p.sum(axis=1,keepdims=True)


def train_mlp(x,y,hidden=12,epochs=100,rate=.1,seed=7,classes=None):
    rng=np.random.default_rng(seed); classes=int(y.max()+1) if classes is None else classes
    w1=rng.normal(0,.15,(x.shape[1],hidden)); b1=np.zeros(hidden); w2=rng.normal(0,.15,(hidden,classes)); b2=np.zeros(classes)
    history=[]; target=np.eye(classes)[y]
    for _ in range(epochs):
        h=np.tanh(x@w1+b1); p=softmax(h@w2+b2); history.append(float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()))
        dz=(p-target)/len(y); dh=(dz@w2.T)*(1-h*h)
        dw2=h.T@dz; db2=dz.sum(axis=0); dw1=x.T@dh; db1=dh.sum(axis=0)
        w1-=rate*dw1; b1-=rate*db1; w2-=rate*dw2; b2-=rate*db2
    return (w1,b1,w2,b2),history


def predict(model,x):
    w1,b1,w2,b2=model; h=np.tanh(x@w1+b1); return softmax(h@w2+b2),h


def windows(seq,width=3):
    # Padding is an additional token, rather than an amino acid.
    tokens='_'+PROTEIN; half=width//2; padded='_'*half+seq+'_'*half
    return np.array([np.eye(len(tokens))[[tokens.index(c) for c in padded[i:i+width]]].ravel() for i in range(len(seq))])


def secondary_data(seed=7,n=60):
    rng=np.random.default_rng(seed); seqs=[]; labels=[]
    pools=['AEKLM','VIFYT','GSPDN']
    for _ in range(n):
        state=int(rng.integers(3)); seq=''.join(rng.choice(list(pools[state]),18)); y=np.full(18,state)
        # Overlapping amino-acid preferences and noise make this a teaching task.
        for j in range(18):
            if rng.random()<.12: seq=seq[:j]+rng.choice(list(PROTEIN))+seq[j+1:]
        seqs.append(seq); labels.append(y)
    return seqs,labels


def masked_data():
    # Entire synthetic sequences are split before extracting context windows.
    motifs=['ACDEFG','KLMNPQ','RSTVWY','AEKLAE','GPGSGP','VIVYFI']
    train=[m*4 for m in motifs]; heldout=[m*3+m[:3] for m in motifs]
    def encode(seqs):
        x=[]; y=[]
        for seq in seqs:
            for i in range(1,len(seq)-1):
                x.append(np.concatenate([np.eye(20)[PROTEIN.index(seq[i-1])],np.eye(20)[PROTEIN.index(seq[i+1])]])); y.append(PROTEIN.index(seq[i]))
        return np.array(x),np.array(y)
    return encode(train),encode(heldout),motifs


def masked_context(left,right):
    return np.concatenate([np.eye(20)[PROTEIN.index(left)],np.eye(20)[PROTEIN.index(right)]])[None,:]
