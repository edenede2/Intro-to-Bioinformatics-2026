"""Streamlit lab views. Results persist across step changes; calculations run on buttons."""
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from . import sequences as sq, hmm, geometry as geom, models
from .datasets import SEARCH_DB, MSA, backbone, prediction_example, pdb_text


def save_run(key,callback):
    if st.button('Run experiment',key=f'run_{key}',type='primary'):
        try:
            st.session_state[f'lab_{key}']=callback()
            st.session_state.pop(f'error_{key}',None)
        except (ValueError,TypeError,KeyError,IndexError,np.linalg.LinAlgError) as exc:
            st.session_state.pop(f'lab_{key}',None)
            st.session_state[f'error_{key}']=str(exc)
    if f'error_{key}' in st.session_state: st.error(st.session_state[f'error_{key}'])
    result=st.session_state.get(f'lab_{key}')
    if result is not None: st.caption('Results reflect the last run. Click Run experiment after changing inputs.')
    return result


def text_input(label,default,key,alphabet=sq.PROTEIN,max_length=100,max_records=1):
    text=st.text_area(label,default,key=key,height=90)
    with st.expander('Optional FASTA upload'):
        upload=st.file_uploader(f'Upload FASTA for {label}',type=['fa','fasta','txt'],key=f'upload_{key}')
    def read():
        raw=text
        if upload is not None:
            if upload.size>100_000: raise ValueError('FASTA file must be below 100 KB.')
            try: raw=upload.getvalue().decode('utf-8')
            except UnicodeError as exc: raise ValueError('FASTA must be UTF-8 text.') from exc
        return sq.parse_fasta(raw,alphabet,max_length,max_records)
    return read


def heatmap(matrix,title,x=None,y=None,path=None,key=None):
    z=np.asarray(matrix,float).copy(); z[~np.isfinite(z)]=np.nan
    f=go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale='Viridis',colorbar=dict(title='value')))
    if path:
        f.add_trace(go.Scatter(x=[p[1] for p in path],y=[p[0] for p in path],mode='lines+markers',line=dict(color='#fb7185',width=3),name='traceback'))
    f.update_layout(title=title,height=420,yaxis=dict(autorange='reversed'))
    st.plotly_chart(f,width='stretch',key=key)


def cloud(named,title='3D coordinates',confidence=None):
    fig=go.Figure()
    for i,(name,p) in enumerate(named.items()):
        marker=dict(size=4)
        if confidence is not None and i==len(named)-1: marker.update(color=confidence,colorscale='RdYlBu',cmin=0,cmax=100,colorbar=dict(title='confidence-like'))
        fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines+markers',name=name,marker=marker))
    fig.update_layout(title=title,height=500,scene=dict(aspectmode='data',xaxis_title='x (Å)',yaxis_title='y (Å)',zaxis_title='z (Å)'))
    st.plotly_chart(fig,width='stretch')


def alignment_text(a,b):
    st.code(a+'\n'+''.join('|' if x==y and x!='-' else ' ' for x,y in zip(a,b))+'\n'+b,language='text')


def biology():
    read=text_input('Coding-strand DNA (5′ → 3′)','ATGGAATTTCCCGGGTAA','bio_seq',sq.DNA,max_length=300)
    frame=st.selectbox('Zero-based reading frame',[0,1,2],key='bio_frame')
    edit=st.selectbox('Mutation type',['Substitution','Insertion before position','Deletion at position'],key='bio_edit')
    position=st.number_input('Mutation position (one-based)',1,300,4,key='bio_position')
    replacement=st.selectbox('Replacement nucleotide',list(sq.DNA),key='bio_replacement')
    def calc():
        seq=read()[0][1]
        if position>len(seq): raise ValueError('Mutation position exceeds sequence length.')
        if edit=='Substitution': mutant=seq[:position-1]+replacement+seq[position:]
        elif edit.startswith('Insertion'): mutant=seq[:position-1]+replacement+seq[position-1:]
        else: mutant=seq[:position-1]+seq[position:]
        def translate(s):
            coding=s[frame:]; coding=coding[:len(coding)//3*3]
            return str(Seq(coding).translate()) if coding else ''
        p,mp=translate(seq),translate(mutant)
        kind='frameshift (single-base insertion/deletion)' if edit!='Substitution' else 'synonymous / unchanged translation' if p==mp else 'nonsense (new stop)' if any(x=='*' and (i>=len(p) or p[i]!='*') for i,x in enumerate(mp)) else 'missense or stop-loss'
        return seq,mutant,p,mp,kind,frame
    r=save_run('biology',calc)
    if r:
        seq,mutant,p,mp,kind,f=r
        st.write('**RNA from coding strand:**',str(Seq(seq).transcribe()))
        st.write('**Reverse complement:**',str(Seq(seq).reverse_complement()))
        st.write('**Translation:**',p or '(no complete codons)')
        st.write('**Mutant DNA:**',mutant,'**Mutant translation:**',mp,'**Effect:**',kind)
        codons=[seq[i:i+3] for i in range(f,len(seq)-2,3)]
        st.dataframe(pd.DataFrame({'codon':codons,'amino acid':list(p)}),hide_index=True)
        st.caption('Standard genetic code; stop markers are retained and partial terminal codons are omitted. Mutations are interpreted in the selected frame; the lab does not infer a real gene or model splicing.')
        st.bar_chart(pd.Series({x:seq.count(x)/len(seq) for x in sq.DNA},name='base fraction'))


def pairwise():
    c1,c2=st.columns(2)
    with c1: ar=text_input('Sequence A','ACDEFG','align_a')
    with c2: br=text_input('Sequence B','ACDYFG','align_b')
    left,right=st.columns(2)
    with left: mode=st.selectbox('Alignment mode',['Global (Needleman–Wunsch)','Local (Smith–Waterman)'])
    with right: scoring=st.selectbox('Substitution scores',['Simple match/mismatch','BLOSUM62','PAM250'])
    c1,c2,c3,c4=st.columns(4)
    with c1: match=st.number_input('Match score',0.,10.,2.)
    with c2: mismatch=st.number_input('Mismatch score',-10.,0.,-1.)
    with c3: go_=st.number_input('Gap opening score',-20.,0.,-2.)
    with c4: ge=st.number_input('Gap extension score',-20.,0.,-1.)
    st.caption('Gap length k costs opening + (k−1) × extension. Equal values give linear gaps.')
    def calc():
        a,b=ar()[0][1],br()[0][1]; matrix=None if scoring.startswith('Simple') else substitution_matrices.load(scoring)
        return sq.align(a,b,match,mismatch,go_,ge,mode.startswith('Local'),matrix),a,b
    r=save_run('pairwise',calc)
    if r:
        result,a,b=r; st.metric('Optimal score',f'{result.score:.3f}'); alignment_text(result.a,result.b)
        if len(result.path)>1:
            step=st.slider('Traceback prefix displayed',1,len(result.path),len(result.path),key='pair_step')
        else: step=1
        heatmap(result.matrix,'Best score over three states',path=result.path[:step])
        state=st.selectbox('Inspect affine DP state',['Match','Gap in B','Gap in A'])
        with st.expander('State matrix and traceback coordinates'):
            st.dataframe(pd.DataFrame(result.states[['Match','Gap in B','Gap in A'].index(state)]))
            st.dataframe(pd.DataFrame(result.path,columns=['A prefix','B prefix','state']))
        st.caption('Grid coordinates are prefix lengths. Pink path is forward order of recovered traceback; states are 0=match, 1=gap in B, 2=gap in A.')


def linear():
    ar=text_input('Sequence A','PLEASANTLY','linear_a',max_length=200)
    br=text_input('Sequence B','MEANLY','linear_b',max_length=200)
    match=st.number_input('Match',0.,10.,2.,key='lin_match'); mismatch=st.number_input('Mismatch',-10.,0.,-1.,key='lin_mismatch'); gap=st.number_input('Linear gap',-10.,0.,-2.,key='lin_gap')
    def calc():
        a,b=ar()[0][1],br()[0][1]; aa,bb,t=sq.hirschberg(a,b,match,mismatch,gap)
        full=sq.align(a,b,match,mismatch,gap)
        return a,b,aa,bb,t,full.score,sq.score_alignment(aa,bb,match,mismatch,gap),sq.rolling_row(a,b,match,mismatch,gap)
    r=save_run('linear',calc)
    if r:
        a,b,aa,bb,t,score,hscore,row=r; alignment_text(aa,bb)
        st.write(f'Full DP score: **{score:g}** · Hirschberg score: **{hscore:g}**')
        st.dataframe(pd.DataFrame({'final rolling row':row}))
        st.table(pd.DataFrame({'method':['Full score grid','Two rolling rows'],'theoretical float64 bytes':[8*(len(a)+1)*(len(b)+1),16*(min(len(a),len(b))+1)]}))
        st.caption('Score-storage comparison only: excludes pointers, Python overhead, output and teaching trace. Affine implementation has three grids.')
        if t:
            step=st.slider('Recursive split',1,len(t),1,key='linear_step'); st.json(t[step-1]); st.dataframe(pd.DataFrame(t))


def matrices():
    selected=st.selectbox('Standard matrix',['BLOSUM62','PAM250'])
    standard=substitution_matrices.load(selected)
    letters=list(sq.PROTEIN); table=pd.DataFrame([[standard[x,y] for y in letters] for x in letters],index=letters,columns=letters)
    with st.expander('Inspect locally bundled standard matrix'): st.dataframe(table)
    st.markdown('### Construct a two-letter matrix (ordered pairs)')
    counts=st.data_editor(pd.DataFrame([[8.,2.],[2.,8.]],index=['A','C'],columns=['A','C']),key='matrix_counts')
    p=st.slider('Background P(A)',.05,.95,.5,.05)
    pseudo=st.number_input('Pseudocount per ordered pair',.01,10.,1.)
    r=save_run('matrices',lambda:sq.log_odds(counts.to_numpy(),[p,1-p],pseudo))
    if r is not None: heatmap(r,'Constructed scores (bits)',x=['A','C'],y=['A','C']); st.dataframe(pd.DataFrame(r,index=['A','C'],columns=['A','C']))


def search():
    qr=text_input('Query protein','ACDEFG','search_query',max_length=60)
    db=text_input('Target database',SEARCH_DB,'search_db',max_length=100,max_records=10)
    method=st.selectbox('Teaching search mode',['BLAST-like X-drop','FASTA-like diagonal filtering','PSI-like profile iterations'])
    k=st.slider('Exact word length',1,5,2); xdrop=st.slider('X-drop',0,10,3)
    inclusion=st.number_input('Profile inclusion score cutoff',0.,100.,6.)
    rounds=st.slider('Profile iterations',1,4,2)
    shuffles=st.slider('Null shuffles (first target)',10,100,30,10)
    kval=st.number_input('Illustrative K (not calibrated)',.001,2.,.1,format='%.3f')
    lam=st.number_input('Illustrative λ (not calibrated)',.01,2.,.5)
    seed=st.number_input('Shuffle random seed',0,999,7)
    st.caption('Synthetic database. FASTA-like mode refines a window around the strongest seed diagonal. PSI-like mode scans ungapped query-width profiles; this is not production PSI-BLAST.')
    def calc():
        query=qr()[0][1]; records=db(); rows=[]; traces=[]
        for name,target in records:
            hit,seeds=sq.seed_search(query,target,k,xdrop=xdrop); exact=sq.align(query,target,local=True)
            heuristic=hit[0]
            if method.startswith('FASTA') and seeds:
                diagonals={d:sum(z[2]==d for z in seeds) for _,_,d in seeds}; diagonal=max(diagonals,key=diagonals.get)
                start=max(0,diagonal-2); end=min(len(target),diagonal+len(query)+2)
                heuristic=sq.align(query,target[start:end],local=True).score
            rows.append({'target':name,'heuristic score':heuristic,'exact local score':exact.score,'seed count':len(seeds),'illustrative E':kval*len(query)*sum(len(t) for _,t in records)*np.exp(-lam*exact.score)})
            traces.extend({'target':name,'query start':i,'target start':j,'diagonal':d} for i,j,d in seeds)
        profiles=[]
        if method.startswith('PSI'):
            profile_rows=[query]
            for turn in range(rounds):
                pssm,_=sq.pssm_from_rows(profile_rows); hits=[(name,*sq.profile_scan(pssm,t)) for name,t in records]
                profile_rows=[query]+[window for name,score,window in hits if score>=inclusion and window!=query]
                profiles.append({'iteration':turn+1,'included rows':len(profile_rows),'hits':hits,'profile':pssm})
        target=records[0][1]; observed=sq.align(query,target,local=True).score; rng=np.random.default_rng(seed)
        null=[sq.align(query,''.join(rng.permutation(list(target))),local=True).score for _ in range(shuffles)]
        p=(1+sum(x>=observed for x in null))/(shuffles+1)
        return rows,traces,profiles,observed,null,p
    r=save_run('search',calc)
    if r:
        rows,traces,profiles,observed,null,p=r; st.dataframe(pd.DataFrame(rows),hide_index=True)
        with st.expander('Seed coordinates'): st.dataframe(pd.DataFrame(traces))
        for entry in profiles:
            st.write(f"Profile round {entry['iteration']}: {entry['included rows']} rows used for next update")
            st.dataframe(pd.DataFrame(entry['hits'],columns=['target','profile score','selected window']))
            heatmap(entry['profile'],f"PSSM used in round {entry['iteration']}",x=list(sq.PROTEIN),key=f"pssm_round_{entry['iteration']}")
        st.metric('Empirical shuffled-target tail probability',f'{p:.3f}')
        fig=px.histogram(x=null,nbins=15,labels={'x':'Null local-alignment score'}); fig.add_vline(x=observed,line_color='red'); st.plotly_chart(fig,width='stretch')
        st.caption('Red line = observed score. Composition-preserving target shuffles define this null; the add-one estimate is not a calibrated homology probability.')


def msa():
    read=text_input('Protein sequences',MSA,'msa_input',max_length=60,max_records=6)
    order_mode=st.selectbox('Progressive order',['Guide-tree order','Input order','Reverse input order'])
    gap=st.number_input('Gap score',-10.,0.,-2.,key='msa_gap')
    def calc():
        records=read()
        if len(records)<2: raise ValueError('Provide at least two sequences.')
        seqs=[s for _,s in records]; n=len(seqs); distance=np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                r=sq.align(seqs[i],seqs[j]); denom=max(len(seqs[i]),len(seqs[j])); identity=sum(x==y and x!='-' for x,y in zip(r.a,r.b))/denom
                distance[i,j]=distance[j,i]=1-identity
        condensed=np.array([distance[i,j] for i in range(n) for j in range(i+1,n)])
        z=linkage(condensed,method='average'); order=list(map(int,leaves_list(z))) if order_mode.startswith('Guide') else list(range(n))
        if order_mode.startswith('Reverse'): order=order[::-1]
        rows,sp,t=sq.progressive(seqs,order,gap=gap); occupancy=[]; heights=[]
        for col in zip(*rows):
            counts=np.array([col.count(x) for x in sq.PROTEIN],float); total=counts.sum(); freq=counts/max(total,1)
            entropy=-sum(p*np.log2(p) for p in freq if p>0); occupancy.append(total/n); heights.append(freq*(np.log2(20)-entropy)*total/n)
        return records,order,rows,sp,t,z,distance,np.array(heights)
    r=save_run('msa',calc)
    if r:
        records,order,rows,sp,t,z,distance,heights=r
        st.metric('Sum-of-pairs score',sp)
        st.code('\n'.join(f'{records[i][0]:18} {row}' for i,row in zip(order,rows)),language='text')
        st.dataframe(pd.DataFrame(t)); heatmap(distance,'Guide distances',x=[x[0] for x in records],y=[x[0] for x in records])
        tree=dendrogram(z,no_plot=True,labels=[name for name,_ in records]); f=go.Figure()
        for xs,ys in zip(tree['icoord'],tree['dcoord']): f.add_trace(go.Scatter(x=xs,y=ys,mode='lines',showlegend=False))
        f.update_layout(title='Average-linkage guide tree',xaxis=dict(tickvals=[5+10*i for i in range(len(records))],ticktext=tree['ivl']),yaxis_title='Guide distance'); st.plotly_chart(f,width='stretch')
        logo=go.Figure()
        for j,letter in enumerate(sq.PROTEIN): logo.add_trace(go.Bar(x=np.arange(1,len(heights)+1),y=heights[:,j],name=letter))
        logo.update_layout(barmode='stack',title='Sequence logo as stacked residue information heights',xaxis_title='Alignment column',yaxis_title='bits × nongap occupancy'); st.plotly_chart(logo,width='stretch')
        st.caption('Bar segments represent letter heights. Gap-gap contributes zero to sum-of-pairs; gaps are excluded from residue frequencies and reduce logo occupancy.')


def hmm_inputs(prefix):
    alphabet='ACGT'
    pi=st.text_input('Initial distribution π','[0.5, 0.5]',key=prefix+'_pi')
    a=st.text_area('Transition matrix A (JSON)','[[0.85,0.15],[0.2,0.8]]',key=prefix+'_a')
    b=st.text_area('Emission matrix B: columns A,C,G,T (JSON)','[[0.4,0.1,0.1,0.4],[0.1,0.4,0.4,0.1]]',key=prefix+'_b')
    obs=st.text_input('Observed sequence','AACGCGTT',key=prefix+'_obs')
    def read():
        seq=sq.parse_fasta(obs,sq.DNA,max_length=150,max_records=1)[0][1]
        try:
            initial,transition,emission=json.loads(pi),json.loads(a),json.loads(b)
            emission=np.asarray(emission,float)
            if emission.ndim!=2 or emission.shape[1]!=4:
                raise ValueError('Emission matrix must have four columns for A,C,G,T.')
            return initial,transition,emission,[alphabet.index(x) for x in seq],seq
        except (ValueError,TypeError) as exc: raise ValueError('Enter numeric JSON arrays with matching dimensions.') from exc
    return read


def hmm_inference():
    read=hmm_inputs('hmm')
    def calc():
        pi,a,b,o,seq=read(); return hmm.infer(pi,a,b,o),seq
    r=save_run('hmm',calc)
    if r:
        result,seq=r; st.metric('Observation log-likelihood',f"{result['log_likelihood']:.6f}")
        st.write('**Viterbi path:**',result['path']); st.write('**Posterior maxima:**',result['posterior'].argmax(axis=1).tolist())
        st.write('**Best path log-probability:**',result['path_log_probability'])
        step=st.slider('Observation position',1,len(seq),1,key='hmm_step') if len(seq)>1 else 1
        st.dataframe(pd.DataFrame({'scaled forward':result['alpha'][step-1],'scaled backward':result['beta'][step-1],'posterior':result['posterior'][step-1],'Viterbi log score':result['viterbi'][step-1]}))
        heatmap(result['posterior'].T,'State posterior across observations',x=[f'{i+1}:{x}' for i,x in enumerate(seq)])


def hmm_learning():
    mode=st.radio('Lab',['Baum–Welch training','Profile-HMM alignment'],horizontal=True)
    if mode.startswith('Baum'):
        read=hmm_inputs('em'); iterations=st.slider('EM iterations',1,40,15)
        def calc():
            pi,a,b,o,seq=read(); return hmm.baum_welch(pi,a,b,o,iterations)
        r=save_run('em',calc)
        if r:
            pi,a,b,history=r; st.line_chart(pd.Series(history,name='log likelihood')); st.write('Learned π',pi); st.write('Learned transitions'); st.dataframe(a); st.write('Learned emissions'); st.dataframe(b)
            st.caption('Single-sequence maximum likelihood without pseudocounts. Unvisited rows retain their prior values; initialization and overfitting matter.')
    else:
        profile=st.text_area('Equal-length profile training rows','ACDEFG\nACDYFG\nACDEYG',key='profile_rows')
        query=st.text_input('Query','ACDKEFG',key='profile_query')
        ins=st.slider('Insertion probability',.01,.4,.1,.01); deletion=st.slider('Deletion probability',.01,.4,.1,.01)
        def calc():
            rows=[s.strip().upper() for s in profile.splitlines() if s.strip()]
            if not 1<=len(rows)<=12 or any(len(x)>60 or set(x)-set(sq.PROTEIN) for x in rows): raise ValueError('Use 1–12 protein rows of length ≤60.')
            _,e=sq.pssm_from_rows(rows); seq=sq.parse_fasta(query,sq.PROTEIN,80,1)[0][1]
            return hmm.profile_viterbi(seq,e,sq.PROTEIN,ins,deletion)
        r=save_run('profile',calc)
        if r:
            score,path,matrix=r; st.metric('Best path log-probability',f'{score:.4f}'); st.dataframe(pd.DataFrame(path)); heatmap(matrix,'Profile boundary / query-prefix Viterbi scores')
            st.caption('M consumes both indices, I consumes query only, D is silent and consumes profile only. Final end transition is included in the displayed path score.')


def pdb_input(prefix):
    upload=st.file_uploader('Optional PDB backbone (first model; C-alpha atoms)',type=['pdb'],key=prefix+'_pdb')
    def read():
        if upload is None: return backbone()
        if upload.size>500_000: raise ValueError('PDB must be below 500 KB.')
        try: return geom.parse_pdb(upload.getvalue().decode('utf-8'))[0]
        except UnicodeError as exc: raise ValueError('PDB must be UTF-8 text.') from exc
    return read


def geometry():
    read=pdb_input('geometry'); axis=st.selectbox('Rotation axis',['x','y','z']); angle=st.slider('Rotation angle (degrees)',-180,180,55)
    translation=st.text_input('Translation [x,y,z] (JSON)','[3,-2,1]')
    second=st.slider('Second rotation about z (degrees)',-180,180,25)
    noise=st.slider('Coordinate noise standard deviation',0.,2.,.1,.05)
    def calc():
        p=read(); v=np.eye(3)[['x','y','z'].index(axis)]; r,quat=geom.quaternion_rotation(v,np.deg2rad(angle)); r2,_=geom.quaternion_rotation([0,0,1],np.deg2rad(second))
        t=np.asarray(json.loads(translation),float)
        if t.shape!=(3,) or not np.isfinite(t).all(): raise ValueError('Translation must have three finite components.')
        q=(p@r+t)@r2+np.random.default_rng(7).normal(0,noise,p.shape); fit=geom.kabsch(p,q)
        return p,q,fit,quat,r@r2,t@r2
    r=save_run('geometry',calc)
    if r:
        p,q,fit,quat,composed,ct=r
        st.write('Axis-angle quaternion [w,x,y,z]',quat); st.write('Composed row-vector rotation'); st.dataframe(composed); st.write('Composed translation',ct)
        st.metric('Raw RMSD',f'{np.sqrt(np.mean(np.sum((p-q)**2,axis=1))):.4f}'); st.metric('Fitted RMSD',f"{fit['rmsd']:.4f}")
        cloud({'source':p,'target':q,'fitted source':fit['fitted']}); st.dataframe(pd.DataFrame({'residue':np.arange(1,len(p)+1),'residual distance':fit['residual']}))
        st.write('Recovered rotation'); st.dataframe(fit['rotation']); st.write('Recovered translation',fit['translation']); st.write('Rotation determinant',np.linalg.det(fit['rotation']))


def distances():
    read=pdb_input('distance'); noise=st.slider('Symmetric distance noise',0.,2.,0.,.1)
    raw=st.text_area('Optional distance matrix (JSON; overrides backbone)','',key='distance_json')
    def calc():
        p=read(); d=cdist(p,p)
        if raw.strip(): d=np.asarray(json.loads(raw),float); p=None
        else:
            n=np.random.default_rng(8).normal(0,noise,d.shape); d=np.maximum(d+(n+n.T)/2,0); np.fill_diagonal(d,0)
        x,e,error=geom.reconstruct(d); return p,d,x,e,error
    r=save_run('distances',calc)
    if r:
        p,d,x,e,error=r; st.metric('Distance reconstruction RMSE',f'{error:.6f}'); st.write('Gram eigenvalues',e); heatmap(d,'Input internal distances')
        named={'reconstructed':x}
        if p is not None: named['original centered']=p-p.mean(axis=0)
        cloud(named)
        st.write('Reflection preserves reconstructed distances:',bool(np.allclose(cdist(x,x),cdist(x*np.array([-1,1,1]),x*np.array([-1,1,1])))))
        if np.min(e)<-1e-7: st.info('Negative eigenvalues indicate a non-Euclidean input. This view uses a best positive three-component approximation.')
        st.download_button('Download reconstructed coordinates',pd.DataFrame(x,columns=['x','y','z']).to_csv(index=False),'reconstructed.csv','text/csv')


def structure():
    read=pdb_input('structure'); target_upload=st.file_uploader('Optional target PDB',type=['pdb'],key='structure_target')
    skip=st.slider('Synthetic target: remove every nth residue (0=none)',0,8,5)
    iterations=st.slider('Correspondence iterations',1,15,6); gap=st.number_input('Correspondence gap score',-10.,-.1,-3.)
    seq_a=st.text_input('Optional sequence A (matches source length)','',key='str_seq_a'); seq_b=st.text_input('Optional sequence B (matches target length)','',key='str_seq_b')
    def calc():
        p=read()
        if target_upload is not None:
            if target_upload.size>500_000: raise ValueError('Target PDB must be below 500 KB.')
            try: q=geom.parse_pdb(target_upload.getvalue().decode('utf-8'))[0]
            except UnicodeError as exc: raise ValueError('PDB must be UTF-8 text.') from exc
        else:
            rotation,_=geom.quaternion_rotation([0,0,1],.4); q=p@rotation+[2,1,0]
            if skip: q=q[[i for i in range(len(q)) if (i+1)%skip]]
        fit,ip,iq,history=geom.structure_align(p,q,iterations,gap)
        if seq_a or seq_b:
            sa=sq.parse_fasta(seq_a,sq.PROTEIN,200,1)[0][1]; sb=sq.parse_fasta(seq_b,sq.PROTEIN,200,1)[0][1]
            if len(sa)!=len(p) or len(sb)!=len(q): raise ValueError('Optional sequences must match their backbone lengths.')
        else: sa='A'*len(p); sb='G'*len(q)
        identity=sum(sa[i]==sb[j] for i,j in zip(ip,iq))/len(ip)
        return p,q,fit,ip,iq,history,identity
    r=save_run('structure',calc)
    if r:
        p,q,fit,ip,iq,history,identity=r
        st.metric('Aligned-pair RMSD',f"{fit['rmsd']:.4f}"); st.write(f'Aligned pairs: {len(ip)} · Source coverage: {len(ip)/len(p):.1%} · Target coverage: {len(iq)/len(q):.1%} · Paired sequence identity: {identity:.1%}')
        cloud({'fitted source':p@fit['rotation']+fit['translation'],'target':q}); st.dataframe(pd.DataFrame(history)); st.dataframe(pd.DataFrame({'source residue (1-based)':np.array(ip)+1,'target residue (1-based)':np.array(iq)+1}))
        st.caption('Default A/G labels deliberately illustrate different sequence labels on related synthetic geometries. This local heuristic is not TM-align.')


@st.cache_data(show_spinner=False)
def train_secondary(hidden,epochs,rate):
    seqs,labels=models.secondary_data(); x=np.concatenate([models.windows(s) for s in seqs[:45]]); y=np.concatenate(labels[:45]); xt=np.concatenate([models.windows(s) for s in seqs[45:]]); yt=np.concatenate(labels[45:])
    model,loss=models.train_mlp(x,y,hidden,epochs,rate,classes=3); p,_=models.predict(model,x); pt,_=models.predict(model,xt)
    return model,loss,float((p.argmax(1)==y).mean()),float((pt.argmax(1)==yt).mean()),yt,pt.argmax(1)


@st.cache_data(show_spinner=False)
def train_masked(hidden,epochs,rate):
    (x,y),(xt,yt),motifs=models.masked_data(); model,loss=models.train_mlp(x,y,hidden,epochs,rate,classes=20); p,_=models.predict(model,xt)
    return model,loss,float((p.argmax(1)==yt).mean()),motifs


def modern():
    mode=st.radio('Experiment',['Secondary-structure network','Structure prediction confidence','Masked residues and embeddings'],horizontal=True)
    st.info('All examples and model training data here are synthetic teaching fixtures. No pretrained biological predictor runs in this app.')
    if mode.startswith('Structure'):
        r=save_run('prediction',prediction_example)
        if r:
            reference,predicted,confidence,pae=r; fitted=geom.kabsch(predicted,reference); cloud({'reference':reference,'synthetic prediction':fitted['fitted']},confidence=confidence)
            st.line_chart(pd.DataFrame({'confidence-like value (0–100)':confidence})); heatmap(pae,'Synthetic PAE-like teaching matrix (Å)')
            st.dataframe(pd.DataFrame({'residue':np.arange(1,len(reference)+1),'confidence-like':confidence,'measured synthetic residual (Å)':fitted['residual']}))
            st.caption('These values were designed for teaching. They are not pLDDT/PAE produced by AlphaFold. Residuals against the synthetic reference are actual geometric differences.')
        return
    hidden=st.slider('Hidden units',4,32,12); epochs=st.slider('Training epochs',20,300,100,20); rate=st.slider('Learning rate',.01,.5,.1,.01)
    if mode.startswith('Secondary'):
        read=text_input('Protein to classify','AEKLMAEKLMGPGSGPVIVYFI','secondary_query',max_length=100)
        def calc():
            seq=read()[0][1]; model,loss,acc,testacc,y,yp=train_secondary(hidden,epochs,rate); probs,_=models.predict(model,models.windows(seq)); return seq,loss,acc,testacc,y,yp,probs
        r=save_run('secondary',calc)
        if r:
            seq,loss,acc,testacc,y,yp,probs=r; st.line_chart(pd.Series(loss,name='training cross-entropy')); st.write(f'Train residue accuracy: {acc:.1%} · Held-out sequence residue accuracy: {testacc:.1%}')
            cm=np.zeros((3,3),int)
            for a,b in zip(y,yp): cm[a,b]+=1
            heatmap(cm,'Held-out confusion matrix (true rows, predicted columns)',x=['H','E','C'],y=['H','E','C'])
            st.dataframe(pd.DataFrame({'residue':list(seq),'predicted class':np.array(['helix','strand','coil'])[probs.argmax(axis=1)],'P(helix)':probs[:,0],'P(strand)':probs[:,1],'P(coil)':probs[:,2]}))
            st.caption('45 synthetic sequences train / 15 held-out sequences test; windows are extracted after the split. These class labels are generated preferences, not experimental secondary structure.')
    else:
        context=st.text_input('Masked context: exactly three symbols, e.g. A?D','A?D')
        def calc():
            s=context.strip().upper()
            if len(s)!=3 or s[1]!='?' or set(s[::2])-set(sq.PROTEIN): raise ValueError('Use one residue, ?, one residue; e.g. A?D.')
            model,loss,acc,motifs=train_masked(hidden,epochs,rate); probs,_=models.predict(model,models.masked_context(s[0],s[2]))
            contexts=[m[i-1]+'?'+m[i+1] for m in motifs for i in range(1,len(m)-1)]
            x=np.concatenate([models.masked_context(s[0],s[2]) for s in contexts]); _,h=models.predict(model,x); centered=h-h.mean(axis=0); u,sv,vt=np.linalg.svd(centered,full_matrices=False); xy=centered@vt[:2].T
            return loss,acc,probs[0],contexts,xy
        r=save_run('masked',calc)
        if r:
            loss,acc,probs,contexts,xy=r; st.line_chart(pd.Series(loss,name='masked-token cross-entropy')); st.metric('Held-out synthetic motif accuracy',f'{acc:.1%}')
            st.dataframe(pd.DataFrame({'residue':list(sq.PROTEIN),'probability':probs}).sort_values('probability',ascending=False),hide_index=True)
            f=px.scatter(x=xy[:,0],y=xy[:,1],text=contexts,labels={'x':'PC1','y':'PC2'},title='Hidden context embeddings projected with PCA'); st.plotly_chart(f,width='stretch')
            st.caption('Two-neighbor context MLP, not a transformer. Held-out sequences share designed motifs with training; this evaluates interpolation, not remote-family generalization.')

LABS={'biology':biology,'pairwise':pairwise,'linear':linear,'matrices':matrices,'search':search,'msa':msa,'hmm':hmm_inference,'hmm_learning':hmm_learning,'geometry':geometry,'distances':distances,'structure':structure,'modern':modern}
