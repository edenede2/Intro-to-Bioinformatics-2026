"""Original questions, grading and versioned portable progress."""
from dataclasses import dataclass, asdict
import json
import math
from datetime import datetime, timezone
from .content import BY_ID

@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    prompt: str
    answer: str | float
    explanation: str
    hint: str
    options: tuple=()
    tolerance: float=.01

# Conceptual, numerical and tracing questions for every module.
RAW={
'biology':[
('Which level describes the amino-acid sequence?','Primary','Primary structure is the chain sequence.','Start from the smallest structural level.',('Primary','Secondary','Tertiary','Quaternary')),
('How many complete codons are in a 14-nucleotide sequence at zero-based frame offset 2?',4.,'floor((14−2)/3)=4.','Remove the frame offset before grouping by three.'),
('What is the reverse complement of 5′-AGTC-3′?','GACT','Complement gives TCAG, then reverse gives GACT.','Apply both operations.',('TCAG','GACT','AGTC','CTGA'))],
'pairwise':[
('Which algorithm can restart at zero?','Smith–Waterman','Local alignment can discard a poor prefix.','Think local rather than global.',('Smith–Waterman','Needleman–Wunsch','Hirschberg')),
('Affine gap score: opening −5, extension −1, length 4?',-8.,'−5+(4−1)×−1=−8.','The first symbol pays opening only.'),
('Global alignment AC/AG: match 2, mismatch −1, gap −2. Optimal score?',1.,'The diagonal gives 2−1=1; inserting gaps is worse.','Compare diagonal with gaps.')],
'linear':[
('What does a rolling-row DP return without extra reconstruction?','Optimal score','Discarded rows do not retain the traceback.','Think about what is stored.',('Optimal score','Full traceback matrix','All optimal alignments')),
('Forward split scores [0,2,1], backward-aligned scores [3,0,4]. Best zero-based split index?',2.,'Combined scores are [3,2,5]; index 2 wins.','Add corresponding entries.'),
('For shorter sequence length 20, how many cells are in two rolling rows?',42.,'Two rows each contain 20+1 border-inclusive entries.','Remember the empty-prefix column.')],
'matrices':[
('An enriched aligned residue pair should have what log-odds sign?','Positive','Observed frequency exceeds independent background.','Is the probability ratio above one?',('Positive','Negative','Always zero')),
('If q=0.5 and background pair probability=0.25, what is log₂ odds?',1.,'log₂(2)=1 bit.','Divide before taking the logarithm.'),
('Which comparison generally represents greater evolutionary distance?','PAM250 rather than PAM30','PAM indices increase with modeled evolutionary distance.','PAM and BLOSUM numbering differ.',('PAM250 rather than PAM30','PAM30 rather than PAM250','Both indices mean identical distance'))],
'search':[
('What is an E-value?','Expected number of chance hits','It counts expected chance hits at least as strong in the stated search space.','It can exceed one.',('Expected number of chance hits','Probability of homology','Fraction of identical residues')),
('Holding score and parameters fixed, E=0.2. What is E after doubling database length?',.4,'E is linear in search-space length.','Use Kmn exp(−λS).'),
('9 of 99 shuffled scores are ≥ the observed score. With add-one correction, empirical p?',.1,'(1+9)/(99+1)=0.1.','Add one to numerator and denominator.')],
'msa':[
('Does a progressive guide tree necessarily establish evolutionary history?','No','It is a heuristic guide and depends on its estimated distances.','Distinguish alignment strategy from phylogenetic inference.',('No','Yes')),
('Column A,A,C; match 2 and mismatch −1. Sum-of-pairs score?',0.,'One identical and two different unordered pairs: 2−1−1=0.','There are three pairs.'),
('Four sequences contribute how many unordered row pairs?',6.,'4×3/2=6.','Count each pair once.')],
'hmm':[
('Which method finds the most probable complete hidden-state path?','Viterbi','Viterbi maximizes joint path probability.','Forward sums rather than maximizes.',('Viterbi','Forward','Posterior marginal argmax')),
('π=[0.6,0.4]; P(A|state)=[0.8,0.2]. What is P(first observation A)?',.56,'0.6×0.8+0.4×0.2=0.56.','Marginalize over initial states.'),
('In Forward, should alternative predecessor paths be summed or maximized?','Summed','Sum computes total evidence over paths.','Compare with Viterbi.',('Summed','Maximized','Discarded'))],
'hmm_learning':[
('Which profile state skips a position without emitting?','Delete','Delete advances profile index without consuming the query.','Contrast insert and match.',('Delete','Insert','Match')),
('Expected transitions [8,2] from one state: updated probability of transition to first state?',.8,'8/(8+2)=0.8.','Normalize expected counts.'),
('Does Baum–Welch guarantee the globally best model?','No','EM improves a local objective and can reach different local optima.','Initialization matters.',('No','Yes'))],
'geometry':[
('A proper rotation has which determinant?',1.,'Orthogonality plus determinant +1 excludes reflection.','Chirality matters.'),
('Identical corresponding points differ only by translation [3,0,0]. RMSD before fitting?',3.,'Every point is separated by distance 3.','Average squared distances and take a square root.'),
('How are unit quaternions q and −q related?','Same rotation','Both encode the same 3D rotation.','The encoding is a double cover.',('Same rotation','Opposite translations','Different chirality'))],
'distances':[
('Can pairwise distances distinguish a shape from its mirror image?','No','Reflection preserves every pairwise distance.','Distance contains no handedness information.',('No','Yes')),
('Points (0,0,0) and (3,4,0) are how far apart?',5.,'sqrt(3²+4²)=5.','Use Euclidean distance.'),
('Large negative centered Gram eigenvalues indicate what?','Non-Euclidean inconsistency','A Euclidean Gram matrix is positive semidefinite.','Think of squared singular values.',('Non-Euclidean inconsistency','A perfect noiseless 3D reconstruction','Known absolute orientation'))],
'structure':[
('What extra task distinguishes structure alignment from superposition?','Finding residue correspondences','Superposition assumes the point pairs are already given.','What does Kabsch require?',('Finding residue correspondences','Changing coordinate units','Removing all gaps')),
('8 of 10 source residues are aligned. Source coverage as a fraction?',.8,'8/10=0.8.','Keep the denominator explicit.'),
('Can selecting fewer pairs lower RMSD while reducing coverage?','Yes','Reporting both metrics prevents misleading comparisons.','Outliers can be excluded.',('Yes','No'))],
'modern':[
('How should overlapping residue windows be separated for evaluation?','Split proteins first','Overlapping windows from one protein can leak information across splits.','Keep related contexts together.',('Split proteins first','Randomly split all windows','Evaluate only on training windows')),
('True class probability is 0.25. Natural-log cross-entropy loss?',1.386294,'−ln(0.25)≈1.386.','Take the negative natural logarithm.'),
('Does high prediction confidence guarantee biological correctness?','No','Confidence is an estimate and needs appropriate validation.','Distinguish estimate from measurement.',('No','Yes'))],
}
QUESTIONS=[Question(f'{topic}_{i}',topic,*row) for topic,rows in RAW.items() for i,row in enumerate(rows)]
QUESTION_BY_ID={q.id:q for q in QUESTIONS}


def grade(q,value):
    if q.options: return value==q.answer
    try: return math.isfinite(float(value)) and abs(float(value)-float(q.answer))<=q.tolerance
    except (TypeError,ValueError): return False


def fresh_progress():
    return {'version':1,'completed':[],'attempts':[]}


def record_attempt(progress,q,value,mode='practice'):
    ok=grade(q,value)
    progress['attempts'].append({'question_id':q.id,'answer':value,'correct':ok,'mode':mode,'time':datetime.now(timezone.utc).isoformat()})
    return ok


def load_progress(raw):
    if len(raw)>1_000_000: raise ValueError('Progress file must be below 1 MB.')
    try: p=json.loads(raw)
    except (ValueError,UnicodeError) as exc: raise ValueError('Progress file must contain valid JSON.') from exc
    if not isinstance(p,dict) or p.get('version')!=1 or set(p)!={'version','completed','attempts'}:
        raise ValueError('Unsupported progress schema; expected version 1.')
    if not isinstance(p['completed'],list) or any(not isinstance(x,str) or x not in BY_ID for x in p['completed']):
        raise ValueError('Unknown completed lesson.')
    if not isinstance(p['attempts'],list) or len(p['attempts'])>10000: raise ValueError('Invalid attempt history.')
    for a in p['attempts']:
        if not isinstance(a,dict) or set(a)!={'question_id','answer','correct','mode','time'} or not isinstance(a['question_id'],str) or a['question_id'] not in QUESTION_BY_ID:
            raise ValueError('Unknown question or invalid attempt.')
        q=QUESTION_BY_ID[a['question_id']]
        if not isinstance(a['answer'],(str,int,float)) or isinstance(a['answer'],bool) or not isinstance(a['correct'],bool) or a['correct']!=grade(q,a['answer']):
            raise ValueError('Attempt score does not match its answer.')
        if not isinstance(a['mode'],str) or a['mode'] not in ('practice','exam') or not isinstance(a['time'],str): raise ValueError('Invalid attempt metadata.')
        try: datetime.fromisoformat(a['time'])
        except ValueError as exc: raise ValueError('Invalid attempt timestamp.') from exc
    p['completed']=list(dict.fromkeys(p['completed']))
    return p

ASSIGNMENTS=[
('Sequence methods', '''1. Align ACDEFG and ACDYFG globally and locally. Predict how a larger gap-opening penalty changes the preferred alignment, then verify it.
2. Recover the global alignment with Hirschberg and explain its split decisions and memory saving.
3. Construct a two-letter log-odds matrix; explain the sign of each entry.
4. Compare seeded search with exact local alignment on a target with one substitution inside each possible seed. Explain a missed hit.
5. Repeat progressive alignment in two orders and compare sum-of-pairs scores.''', '''A complete solution states the scoring convention, displays matrices/tracebacks, checks the Hirschberg score against full DP, normalizes counts, and distinguishes heuristic recall from optimality. Alignment order can change an MSA even when all pairwise steps are optimal.'''),
('Hidden Markov models', '''1. For the default HMM and observations ACG, compute the first Forward row by hand.
2. Compare the Viterbi path with posterior state maxima; explain why they need not agree.
3. Train with Baum–Welch from two initializations. Plot likelihood and explain any difference in final parameters.
4. Build a profile from three short rows and align a query containing both an insertion and a deletion. Identify which steps consume a query symbol.''', '''The initial row is π multiplied by first-symbol emissions before scaling. Use sums for Forward and maxima for Viterbi. Likelihood improvement does not guarantee a global optimum. M consumes both indices, I consumes the query, and D consumes the profile only.'''),
('Structure and learning', '''1. Rotate and translate a synthetic backbone, then recover its proper rigid transform. Compare raw and fitted RMSD.
2. Reconstruct it from pairwise distances. Explain orientation and reflection ambiguity using a mirror image.
3. Delete several target residues, perform structural alignment, and report RMSD plus both coverage fractions.
4. Train the secondary-structure classifier and compare training with held-out sequence performance.
5. Explain what the synthetic confidence and masked-token experiments can and cannot establish about natural proteins.''', '''A proper rotation has determinant +1. Distances recover shape but not handedness. Report pair counts with RMSD. Split whole sequences before creating windows. Synthetic confidence values and small motif models demonstrate mechanics, not validated biological prediction.'''),
]
