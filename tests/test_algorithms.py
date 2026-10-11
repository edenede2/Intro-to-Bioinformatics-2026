import itertools
import json
import numpy as np
import pytest
from Bio.Align import PairwiseAligner
from scipy.spatial.distance import cdist
from bioapp import sequences as sq, hmm, geometry as geo, models
from bioapp.datasets import backbone, pdb_text
from bioapp.assessment import QUESTIONS, fresh_progress, record_attempt, load_progress
from bioapp.content import LESSONS


@pytest.mark.parametrize('local',[False,True])
@pytest.mark.parametrize('opening,extension',[(-2,-2),(-4,-1),(-1,-3)])
def test_affine_against_biopython(local,opening,extension):
    reference=PairwiseAligner(); reference.mode='local' if local else 'global'
    reference.match_score=2; reference.mismatch_score=-1
    reference.open_gap_score=opening; reference.extend_gap_score=extension
    strings=['A','C','AC','CA','AAC','ACC','CAC']
    for a,b in itertools.product(strings,repeat=2):
        result=sq.align(a,b,2,-1,opening,extension,local)
        assert result.score==pytest.approx(reference.score(a,b))
        assert len(result.a)==len(result.b)
        score=0; previous=None
        for x,y in zip(result.a,result.b):
            state=1 if y=='-' else 2 if x=='-' else 0
            score+=(extension if previous==state else opening) if state else 2 if x==y else -1
            previous=state
        assert score==pytest.approx(result.score)
        if not local:
            assert result.a.replace('-','')==a
            assert result.b.replace('-','')==b


def test_hirschberg_and_empty_inputs():
    strings=['','A','C','AC','CA','AAC','ACACA']
    for a,b in itertools.product(strings,repeat=2):
        aa,bb,trace=sq.hirschberg(a,b)
        assert aa.replace('-','')==a and bb.replace('-','')==b
        score=sq.align(a,b).score
        assert sq.score_alignment(aa,bb)==score
        assert sq.rolling_row(a,b)[-1]==score
    assert sq.align('A','C',local=True).score==0


def test_sequence_validation_and_profiles():
    assert sq.parse_fasta('>a\nac\n>c\nGT',sq.DNA)==[('a','AC'),('c','GT')]
    for text in ['', '>a', '>a\n>c\nAC','ACN']:
        with pytest.raises(ValueError): sq.parse_fasta(text,sq.DNA)
    with pytest.raises(ValueError): sq.align('A'*401,'A')
    with pytest.raises(ValueError): sq.log_odds([[1,2],[3,4]],[.4,.4])
    scores=sq.log_odds([[8,2],[2,8]],[.5,.5])
    assert scores[0,0]>0 and scores[0,1]<0
    rows,sp,trace=sq.progressive(['ACD','AD','ACED'])
    assert len({len(r) for r in rows})==1
    assert [r.replace('-','') for r in rows]==['ACD','AD','ACED']
    hit,seeds=sq.seed_search('ACDE','XXACDEYY',2)
    assert hit[0]==8 and len(seeds)==3
    assert sq.seed_search('AC','GT',2)[0][0]==0
    pssm,p=sq.pssm_from_rows(['AC','AC']); assert np.allclose(p.sum(1),1)
    assert sq.profile_scan(pssm,'AC')[0]>sq.profile_scan(pssm,'WW')[0]


def test_hmm_exhaustive():
    pi=np.array([.6,.4]); a=np.array([[.7,.3],[.2,.8]]); b=np.array([[.8,.2],[.3,.7]]); o=[0,1,0,1]
    r=hmm.infer(pi,a,b,o); paths=list(itertools.product(range(2),repeat=len(o))); weights=[]
    for path in paths:
        w=pi[path[0]]*b[path[0],o[0]]
        for t in range(1,len(o)): w*=a[path[t-1],path[t]]*b[path[t],o[t]]
        weights.append(w)
    evidence=sum(weights)
    assert np.exp(r['log_likelihood'])==pytest.approx(evidence)
    assert tuple(r['path'])==paths[int(np.argmax(weights))]
    assert np.exp(r['path_log_probability'])==pytest.approx(max(weights))
    posterior=np.zeros((len(o),2))
    for path,w in zip(paths,weights):
        for t,s in enumerate(path): posterior[t,s]+=w/evidence
    assert np.allclose(r['posterior'],posterior)
    pi2,a2,b2,history=hmm.baum_welch(pi,a,b,o,20)
    assert np.all(np.diff(history)>=-1e-10)
    assert a2.sum(1)==pytest.approx([1,1]); assert b2.sum(1)==pytest.approx([1,1])


def test_hmm_failure_and_single_observation():
    with pytest.raises(ValueError): hmm.infer([1],[[1]],[[1,0]],[1])
    with pytest.raises(ValueError): hmm.infer([.4,.4],[[1,0],[0,1]],[[.5,.5],[.5,.5]],[0])
    with pytest.raises(ValueError): hmm.infer([1],[[1]],[[1]],[])
    pi,a,b,h=hmm.baum_welch([1],[[1]],[[.5,.5]],[0],3)
    assert np.allclose(a,[[1]]) and h[-1]==0


def test_profile_path_consumption():
    alphabet='AC'; emissions=np.array([[.99,.01],[.01,.99],[.99,.01]])
    for seq in ['ACA','ACCA','AA']:
        score,path,d=hmm.profile_viterbi(seq,emissions,alphabet,.1,.1)
        assert np.isfinite(score)
        assert sum(x['state'] in ('M','D') for x in path)==3
        assert sum(x['state'] in ('M','I') for x in path)==len(seq)
    assert 'D' in [x['state'] for x in hmm.profile_viterbi('AA',emissions,alphabet)[1]]
    assert 'I' in [x['state'] for x in hmm.profile_viterbi('ACCA',emissions,alphabet)[1]]


def test_geometry_recovery_reflection_and_reconstruction():
    rng=np.random.default_rng(2); p=rng.normal(size=(12,3)); r,q=geo.quaternion_rotation([1,2,3],.7); t=np.array([3,-1,2]); target=p@r+t
    fit=geo.kabsch(p,target)
    assert fit['rmsd']<1e-12
    assert np.allclose(fit['rotation'],r) and np.allclose(fit['translation'],t)
    assert np.linalg.norm(q)==pytest.approx(1)
    mirror=p*np.array([-1,1,1]); fit=geo.kabsch(p,mirror)
    assert np.linalg.det(fit['rotation'])==pytest.approx(1)
    assert fit['rmsd']>.1
    d=cdist(p,p); x,e,error=geo.reconstruct(d)
    assert error<1e-12 and np.allclose(cdist(x,x),d)
    assert np.allclose(cdist(p,p),cdist(mirror,mirror))
    with pytest.raises(ValueError): geo.kabsch([[0,0,0],[1,0,0],[2,0,0]],[[0,0,0],[1,0,0],[2,0,0]])
    with pytest.raises(ValueError): geo.reconstruct([[0,1,1],[2,0,1],[1,1,0]])
    with pytest.raises(ValueError): geo.quaternion_rotation([0,0,0],1)
    with pytest.raises(ValueError): geo.parse_pdb('not a PDB')
    restored,names=geo.parse_pdb(pdb_text(backbone()))
    assert np.allclose(restored,backbone(),atol=.001)
    target=backbone()@r+t
    fit,ip,iq,h=geo.structure_align(backbone(),target)
    assert fit['rmsd']<1e-10 and len(ip)==24
    assert all(np.diff(ip)>0) and all(np.diff(iq)>0)


def test_mlp_loss_and_probabilities():
    x=np.eye(3).repeat(10,axis=0); y=np.arange(3).repeat(10)
    model,loss=models.train_mlp(x,y,epochs=100,rate=.3,classes=3)
    p,h=models.predict(model,x)
    assert loss[-1]<loss[0]
    assert np.allclose(p.sum(1),1) and (p.argmax(1)==y).mean()>.9


def test_progress_roundtrip_and_rejections():
    p=fresh_progress(); q=QUESTIONS[0]; record_attempt(p,q,q.answer)
    p['completed']=['biology']; assert load_progress(json.dumps(p))==p
    for raw in ['{}','null','not json',json.dumps({**p,'version':2}),json.dumps({**p,'completed':['unknown']})]:
        with pytest.raises(ValueError): load_progress(raw)
    corrupt=json.loads(json.dumps(p)); corrupt['attempts'][0]['correct']=False
    with pytest.raises(ValueError): load_progress(json.dumps(corrupt))
    for l in LESSONS:
        assert len(l.notes)>500 and l.formula and l.worked and l.sources
        qs=[q for q in QUESTIONS if q.topic==l.id]
        assert len(qs)>=3 and any(not q.options for q in qs)


@pytest.mark.parametrize('initial',[.5,[[.5,.5]],[],[1]*9])
def test_malformed_hmm_shapes(initial):
    with pytest.raises(ValueError): hmm.infer(initial,[[1]],[[1]],[0])


def test_profile_normalization_and_corrupt_progress():
    with pytest.raises(ValueError): hmm.profile_viterbi('A',[[.5,.4]],'AC')
    p=fresh_progress(); q=QUESTIONS[0]; record_attempt(p,q,q.answer)
    p['attempts'][0]['question_id']=[]
    with pytest.raises(ValueError): load_progress(json.dumps(p))
    with pytest.raises(ValueError): hmm.infer([1],[[1]],[[.5,.5]],[.5])
