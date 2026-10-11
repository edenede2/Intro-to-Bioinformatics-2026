"""Inspectable sequence algorithms. Gap cost = open + (length-1)*extend."""
from dataclasses import dataclass
import numpy as np

DNA = 'ACGT'
PROTEIN = 'ACDEFGHIKLMNPQRSTVWY'


def parse_fasta(text, alphabet=PROTEIN, max_length=400, max_records=12):
    records, name, chunks = [], 'sequence_1', []
    header_seen = False
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if chunks:
                records.append((name, ''.join(chunks).upper()))
            elif header_seen:
                raise ValueError('A FASTA record has no sequence.')
            name, chunks = line[1:].strip() or f'sequence_{len(records)+1}', []
            header_seen = True
        else:
            chunks.append(''.join(line.split()))
    if chunks:
        records.append((name, ''.join(chunks).upper()))
    else:
        raise ValueError('Enter a nonempty sequence for every FASTA record.')
    if len(records) > max_records:
        raise ValueError(f'Use at most {max_records} sequences.')
    for name, seq in records:
        if not seq or len(seq) > max_length:
            raise ValueError(f'{name}: length must be 1–{max_length}.')
        invalid = set(seq) - set(alphabet)
        if invalid:
            raise ValueError(f'{name}: unsupported symbols {sorted(invalid)}; expected {alphabet}.')
    return records


@dataclass
class Alignment:
    score: float
    a: str
    b: str
    matrix: np.ndarray
    path: list
    states: np.ndarray


def align(a, b, match=2., mismatch=-1., gap_open=-2., gap_extend=None, local=False, substitution=None):
    """Three-state DP: match, gap in b, gap in a. Deterministic traceback."""
    if len(a) > 400 or len(b) > 400:
        raise ValueError('Alignment input is limited to 400 symbols per sequence.')
    ge = gap_open if gap_extend is None else gap_extend
    if gap_open > 0 or ge > 0:
        raise ValueError('Gap scores must be nonpositive.')
    n, m = len(a), len(b)
    d = np.full((3, n+1, m+1), -np.inf)
    ptr = np.full((3, n+1, m+1), -1, dtype=int)
    d[0, 0, 0] = 0
    if local:
        d[:, 0, :] = 0; d[:, :, 0] = 0
    else:
        for i in range(1, n+1):
            d[1, i, 0] = gap_open + (i-1)*ge; ptr[1, i, 0] = 1 if i > 1 else 0
        for j in range(1, m+1):
            d[2, 0, j] = gap_open + (j-1)*ge; ptr[2, 0, j] = 2 if j > 1 else 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            s = float(substitution[a[i-1], b[j-1]]) if substitution is not None else (match if a[i-1] == b[j-1] else mismatch)
            candidates = [d[:, i-1, j-1] + s,
                          d[:, i-1, j] + [gap_open, ge, gap_open],
                          d[:, i, j-1] + [gap_open, gap_open, ge]]
            for state, values in enumerate(candidates):
                k = int(np.argmax(values)); v = values[k]
                if local and v <= 0:
                    d[state, i, j] = 0
                else:
                    d[state, i, j] = v; ptr[state, i, j] = k
    if local:
        state, i, j = np.unravel_index(np.argmax(d), d.shape)
    else:
        i, j = n, m; state = int(np.argmax(d[:, i, j]))
    score, aa, bb, path = float(d[state, i, j]), [], [], []
    while i or j:
        if local and d[state, i, j] <= 0:
            break
        path.append((int(i), int(j), int(state)))
        prev = ptr[state, i, j]
        if state == 0:
            aa.append(a[i-1]); bb.append(b[j-1]); i -= 1; j -= 1
        elif state == 1:
            aa.append(a[i-1]); bb.append('-'); i -= 1
        else:
            aa.append('-'); bb.append(b[j-1]); j -= 1
        state = int(prev)
    path.append((int(i), int(j), 0))
    return Alignment(score, ''.join(aa[::-1]), ''.join(bb[::-1]), d.max(axis=0), path[::-1], d)


def score_alignment(a, b, match=2, mismatch=-1, gap=-2):
    return sum(0 if x == y == '-' else gap if '-' in (x,y) else match if x == y else mismatch for x,y in zip(a,b))


def rolling_row(a, b, match=2, mismatch=-1, gap=-2):
    row = np.arange(len(b)+1, dtype=float)*gap
    for i,x in enumerate(a, 1):
        new = np.empty(len(b)+1); new[0] = i*gap
        for j,y in enumerate(b, 1):
            new[j] = max(row[j-1]+(match if x == y else mismatch), row[j]+gap, new[j-1]+gap)
        row = new
    return row


def hirschberg(a,b,match=2,mismatch=-1,gap=-2):
    """Linear working space in the shorter sequence; output/trace excluded."""
    swapped = len(b) > len(a)
    if swapped: a,b = b,a
    trace = []
    def solve(x,y,depth):
        if not x: return '-'*len(y),y
        if not y: return x,'-'*len(x)
        if len(x) == 1 or len(y) == 1:
            r=align(x,y,match,mismatch,gap); return r.a,r.b
        mid=len(x)//2
        totals=rolling_row(x[:mid],y,match,mismatch,gap)+rolling_row(x[mid:][::-1],y[::-1],match,mismatch,gap)[::-1]
        split=int(totals.argmax())
        trace.append({'depth':depth,'a_length':len(x),'b_length':len(y),'a_split':mid,'b_split':split,'best_score':float(totals[split])})
        # Release rows before recursion to retain linear working space.
        del totals
        l1,l2=solve(x[:mid],y[:split],depth+1)
        r1,r2=solve(x[mid:],y[split:],depth+1)
        return l1+r1,l2+r2
    aa,bb=solve(a,b,0)
    if swapped: aa,bb=bb,aa
    return aa,bb,trace


def log_odds(counts, background, pseudocount=1):
    c=np.asarray(counts,dtype=float); p=np.asarray(background,dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or p.shape != (c.shape[0],):
        raise ValueError('Counts must be square and match the background vector.')
    if not np.isfinite(c).all() or (c<0).any() or not np.isfinite(p).all() or (p<=0).any() or not np.isclose(p.sum(),1):
        raise ValueError('Counts must be finite and nonnegative; positive background probabilities must sum to one.')
    if pseudocount <= 0: raise ValueError('Use a positive pseudocount.')
    q=(c+pseudocount); q/=q.sum()
    return np.log2(q/np.outer(p,p))


def progressive(seqs, order=None, match=2, mismatch=-1, gap=-2):
    """Profile-to-sequence DP with summed residue scores (linear gaps)."""
    order=list(range(len(seqs))) if order is None else order
    profile=[seqs[order[0]]]; trace=[]
    for index in order[1:]:
        seq=seqs[index]; n=len(profile[0]); m=len(seq); k=len(profile)
        d=np.zeros((n+1,m+1)); ptr=np.zeros_like(d,dtype=int)
        for i in range(1,n+1): d[i,0]=d[i-1,0]+sum(gap if p[i-1]!='-' else 0 for p in profile); ptr[i,0]=1
        d[0,:]=np.arange(m+1)*k*gap; ptr[0,1:]=2
        for i in range(1,n+1):
            col=[p[i-1] for p in profile]
            for j in range(1,m+1):
                vals=[d[i-1,j-1]+sum(gap if x=='-' else match if x==seq[j-1] else mismatch for x in col),
                      d[i-1,j]+sum(gap if x!='-' else 0 for x in col),d[i,j-1]+k*gap]
                ptr[i,j]=np.argmax(vals); d[i,j]=max(vals)
        rows=[[] for _ in range(k+1)]; i,j=n,m
        while i or j:
            state=ptr[i,j]
            for z in range(k): rows[z].append(profile[z][i-1] if state!=2 else '-')
            rows[k].append(seq[j-1] if state!=1 else '-')
            if state!=2: i-=1
            if state!=1: j-=1
        profile=[''.join(r[::-1]) for r in rows]
        trace.append({'added_sequence':index+1,'profile_score':float(d[n,m]),'columns':len(profile[0])})
    sp=sum(score_alignment(profile[i],profile[j],match,mismatch,gap) for i in range(len(profile)) for j in range(i))
    return profile,sp,trace


def seed_search(query,target,k=2,match=2,mismatch=-1,xdrop=3):
    """Exact-word BLAST-like ungapped X-drop extensions, not NCBI BLAST."""
    seeds=[]; best=(0,'','')
    for i in range(len(query)-k+1):
        for j in range(len(target)-k+1):
            if query[i:i+k]!=target[j:j+k]: continue
            seeds.append((i,j,j-i)); left=right=0; gain=peak=0
            for step in range(1,min(i,j)+1):
                gain+=match if query[i-step]==target[j-step] else mismatch
                if gain>peak: peak=gain; left=step
                if peak-gain>xdrop: break
            left_score=peak; gain=peak=0
            for step in range(min(len(query)-i-k,len(target)-j-k)):
                gain+=match if query[i+k+step]==target[j+k+step] else mismatch
                if gain>peak: peak=gain; right=step+1
                if peak-gain>xdrop: break
            score=k*match+left_score+peak
            if score>best[0]: best=(score,query[i-left:i+k+right],target[j-left:j+k+right])
    return best,seeds


def pssm_from_rows(rows, alphabet=PROTEIN, pseudocount=1):
    if not rows or len({len(r) for r in rows}) != 1: raise ValueError('Profile rows must have equal lengths.')
    c=np.full((len(rows[0]),len(alphabet)),float(pseudocount))
    for row in rows:
        for i,x in enumerate(row):
            if x in alphabet: c[i,alphabet.index(x)]+=1
    p=c/c.sum(axis=1,keepdims=True)
    return np.log2(p*len(alphabet)),p


def profile_scan(pssm,seq,alphabet=PROTEIN):
    w=len(pssm)
    scores=[sum(pssm[i,alphabet.index(x)] for i,x in enumerate(seq[j:j+w])) for j in range(len(seq)-w+1)]
    if not scores: return float('-inf'),''
    j=int(np.argmax(scores)); return float(scores[j]),seq[j:j+w]
