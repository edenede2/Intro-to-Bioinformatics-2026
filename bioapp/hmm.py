"""Scaled discrete HMM inference and EM; profile HMM with silent deletions."""
import numpy as np


def validate(pi,a,b,obs):
    pi,a,b=np.asarray(pi,float),np.asarray(a,float),np.asarray(b,float)
    if pi.ndim!=1 or not 1<=len(pi)<=8:
        raise ValueError('Use an initial vector with 1–8 states.')
    s=len(pi)
    if a.shape!=(s,s) or b.ndim!=2 or b.shape[0]!=s:
        raise ValueError('HMM matrix dimensions do not agree.')
    for x in (pi,a,b):
        if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(axis=-1),1):
            raise ValueError('Probabilities must be finite, nonnegative, and each row must sum to 1.')
    raw=np.asarray(obs)
    if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.number) or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
        raise ValueError('Observations must be a one-dimensional array of integer symbol indices.')
    obs=np.asarray(raw,int)
    if not len(obs) or len(obs)>500 or (obs<0).any() or (obs>=b.shape[1]).any():
        raise ValueError('Use 1–500 observations from the emission alphabet.')
    return pi,a,b,obs


def infer(pi,a,b,obs):
    pi,a,b,o=validate(pi,a,b,obs); t=len(o); s=len(pi)
    alpha=np.zeros((t,s)); scales=np.zeros(t)
    alpha[0]=pi*b[:,o[0]]; scales[0]=alpha[0].sum()
    if scales[0]<=0: raise ValueError('This observation sequence is impossible under the model.')
    alpha[0]/=scales[0]
    for i in range(1,t):
        alpha[i]=(alpha[i-1]@a)*b[:,o[i]]; scales[i]=alpha[i].sum()
        if scales[i]<=0: raise ValueError('This observation sequence is impossible under the model.')
        alpha[i]/=scales[i]
    beta=np.ones((t,s))
    for i in range(t-2,-1,-1): beta[i]=a@(b[:,o[i+1]]*beta[i+1])/scales[i+1]
    gamma=alpha*beta; gamma/=gamma.sum(axis=1,keepdims=True)
    with np.errstate(divide='ignore'): la=np.log(a); lb=np.log(b); lp=np.log(pi)
    v=np.full((t,s),-np.inf); ptr=np.zeros((t,s),int); v[0]=lp+lb[:,o[0]]
    for i in range(1,t):
        candidates=v[i-1][:,None]+la
        ptr[i]=candidates.argmax(axis=0); v[i]=candidates.max(axis=0)+lb[:,o[i]]
    path=[int(v[-1].argmax())]
    for i in range(t-1,0,-1): path.append(int(ptr[i,path[-1]]))
    return {'alpha':alpha,'beta':beta,'posterior':gamma,'log_likelihood':float(np.log(scales).sum()),'path':path[::-1],'viterbi':v,'path_log_probability':float(v[-1].max())}


def baum_welch(pi,a,b,obs,iterations=20):
    pi,a,b,o=validate(pi,a,b,obs); history=[]
    for step in range(iterations+1):
        r=infer(pi,a,b,o); history.append(r['log_likelihood'])
        if step==iterations: break
        gamma=r['posterior']; xi=np.zeros_like(a)
        for t in range(len(o)-1):
            x=r['alpha'][t,:,None]*a*(b[:,o[t+1]]*r['beta'][t+1])[None,:]
            xi+=x/x.sum()
        pi=gamma[0].copy()
        den=xi.sum(axis=1,keepdims=True)
        a=np.divide(xi,den,out=a.copy(),where=den>0)
        counts=np.zeros_like(b)
        for symbol in range(b.shape[1]): counts[:,symbol]=gamma[o==symbol].sum(axis=0)
        den=counts.sum(axis=1,keepdims=True)
        b=np.divide(counts,den,out=b.copy(),where=den>0)
    return pi,a,b,history


def profile_viterbi(seq, emissions, alphabet, insert=0.1, delete=0.1):
    """Small generative profile: consume a match/delete or insert at each boundary.
    Every boundary offers I with insert, D with delete, M with 1-I-D.
    Final boundary offers I with insert or end with 1-insert.
    """
    if not (0<insert<1 and 0<delete<1 and insert+delete<1): raise ValueError('Insertion + deletion probability must be below 1.')
    e=np.asarray(emissions,float)
    if e.ndim!=2 or e.shape[1]!=len(alphabet) or not 1<=len(e)<=100 or len(seq)>200 or not np.isfinite(e).all() or (e<=0).any() or not np.allclose(e.sum(1),1):
        raise ValueError('Use 1–100 normalized, strictly positive profile emission rows and a query of at most 200 symbols.')
    n=len(e); m=len(seq)
    if set(seq)-set(alphabet): raise ValueError('Query contains unsupported symbols.')
    d=np.full((n+1,m+1),-np.inf); ptr={}; d[0,0]=0
    for i in range(n+1):
        for j in range(m+1):
            if i==0 and j==0: continue
            candidates=[]
            if i and j: candidates.append((d[i-1,j-1]+np.log(1-insert-delete)+np.log(e[i-1,alphabet.index(seq[j-1])]),i-1,j-1,'M'))
            if i: candidates.append((d[i-1,j]+np.log(delete),i-1,j,'D'))
            if j: candidates.append((d[i,j-1]+np.log(insert)-np.log(len(alphabet)),i,j-1,'I'))
            best=max(candidates,key=lambda x:x[0]); d[i,j]=best[0]; ptr[i,j]=best[1:]
    i,j=n,m; path=[]
    while i or j:
        ni,nj,state=ptr[i,j]; path.append({'state':state,'profile_position':i,'query_position':j}); i,j=ni,nj
    return float(d[n,m]+np.log(1-insert)),path[::-1],d
