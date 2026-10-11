"""Syllabus-led lessons; original explanations and exercises, not lecture transcripts."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    prerequisites: str
    motivation: str
    notes: str
    formula: str
    symbols: str
    worked: str
    pitfalls: str
    complexity: str
    pseudocode: str
    sources: tuple

BIO=('NCBI: Molecular Biology of the Cell','https://www.ncbi.nlm.nih.gov/books/NBK21054/')
ALIGN=('Biopython pairwise alignment reference','https://biopython.org/docs/latest/Tutorial/chapter_pairwise.html')
HIRSCH=('Hirschberg: A linear space algorithm','https://doi.org/10.1145/360825.360861')
BLAST=('NCBI: sequence similarity statistics','https://blast.ncbi.nlm.nih.gov/tutorial/Altschul-1.html')
PSI=('NCBI: iterated profile searches','https://blast.ncbi.nlm.nih.gov/tutorial/Altschul-2.html')
HMM=('Rabiner: A tutorial on hidden Markov models','https://doi.org/10.1109/5.18626')
KABSCH=('Kabsch: optimal rotation of two vectors','https://doi.org/10.1107/S0567739476001873')
MDS=('SciPy: classical multidimensional scaling foundations','https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html')
AF=('EMBL-EBI: interpreting prediction confidence','https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/')
ESM=('Lin et al.: language models and protein structure','https://doi.org/10.1126/science.ade2574')

LESSONS=[
Lesson('biology','Biology foundations','None; start here.',
'Sequences encode molecules, but the computational representation must preserve direction, alphabet, and biological meaning.',
'''DNA uses A, C, G, T; RNA uses U in place of T. Complementary DNA strands run in opposite directions. Transcription produces RNA from a template; when starting from the coding strand, replace T with U. Translation reads RNA codons in a selected 5′→3′ reading frame. The genetic code is redundant: different codons can encode the same amino acid.

Proteins are chains of amino acids. Primary structure is sequence; secondary structure includes helices and sheets; tertiary structure is the folded chain; quaternary structure describes assemblies. Hydrogen bonding, hydrophobic packing, electrostatics, and steric constraints contribute to structure. A sequence edit may be synonymous, missense, nonsense, or a frameshift, depending on its context.''',
 r'N_{\mathrm{complete\ codons}}=\left\lfloor\frac{L-f}{3}\right\rfloor',
'L is sequence length; f is the zero-based frame offset (0, 1, or 2).',
'ATGGAA TAA in frame 0 gives AUG GAA UAA: methionine, glutamate, stop. GAA→GAG is synonymous; GAA→TAA introduces a stop.',
'A complement is not a reverse complement. A codon list is not evidence of an expressed gene. Ignore neither partial terminal codons nor stop symbols.',
'Sequence inspection, reverse complement, and translation are O(L).',
'validate alphabet\nreverse complement = reverse(complement(sequence))\nfor each complete codon in chosen frame:\n    look up amino acid; retain stop markers', (BIO,)),
Lesson('pairwise','Pairwise sequence alignment','Biology; arrays and maximum operations.',
'Find a scoring-optimal correspondence while allowing substitutions, insertions, and deletions.',
'''Needleman–Wunsch aligns complete sequences. Initialize its borders with accumulated gap costs. Smith–Waterman finds a local segment: borders start at zero, every cell can restart at zero, and traceback starts at the largest score and stops at zero.

A linear gap costs g per missing symbol. An affine gap of length k costs o+(k−1)e, favoring one long gap over many openings when opening is more costly. Three DP states distinguish a residue pair, a gap in the second sequence, and a gap in the first. Traceback follows the predecessor that produced each score; tied optima can yield different valid alignments.''',
 r'F_{ij}=\max(F_{i-1,j-1}+s(a_i,b_j),F_{i-1,j}+g,F_{i,j-1}+g)',
'F is the best prefix score; s is the substitution score; g≤0 is a linear gap score. Local alignment also includes 0 in the maximum.',
'With match 2, mismatch −1, gap −2: AC against AG has diagonal score 2−1=1. Local alignment keeps A/A for score 2.',
'Optimal score does not establish homology. State the gap convention. Do not initialize global borders as zeros.',
'O(nm) time and O(nm) storage; affine alignment stores three grids.',
'initialize borders\nfor i, j in row order:\n    calculate match / insertion / deletion states\nchoose endpoint (bottom-right or global maximum)\nfollow predecessors to recover alignment', (ALIGN,)),
Lesson('linear','Linear-space alignment','Global pairwise alignment; recursion.',
'Compute exact global alignments when a full traceback matrix is too large.',
'''Only the preceding row is needed to compute the next score row. This gives the optimal score but discards traceback. Hirschberg restores an alignment by divide and conquer: cut the first sequence in half, compute forward prefix scores and backward suffix scores, and choose the second-sequence cut maximizing their sum.

Recurse on the two independent subproblems. Empty strings and one-symbol inputs are base cases. The lab uses linear gaps; substituting an affine model requires additional split-state bookkeeping.''',
 r'j^*=\arg\max_j\{L_j+R_{m-j}\}',
'L scores the left half against prefixes of b; R scores the reversed right half against reversed suffixes.',
'For AC/AC, cutting a after A gives forward [−2,2,0] and reverse-combined [0,2,−2]; split j=1 yields 4.',
'Reverse both suffix strings. Different tie choices need not produce the same optimal alignment. Output and teaching traces also consume memory.',
'O(nm) time; O(min(n,m)) DP working rows, excluding inputs, recursion bookkeeping, alignment output, and optional trace.',
'if either string empty: return gaps\nif one symbol: solve directly\ncompute forward and backward rows\nchoose maximum combined split\nrelease rows; recurse; concatenate', (HIRSCH,ALIGN)),
Lesson('matrices','Substitution matrices','Probability; logarithms; alignment scoring.',
'Common evolutionary changes should carry different evidence than rare or implausible changes.',
'''A log-odds matrix compares an observed aligned-pair distribution with a background independent-pair distribution. Positive scores favor enriched pairs; negative scores penalize depleted pairs. Pseudocounts prevent zero probabilities and stabilize small samples. The construction lab uses ordered pairs with a user-specified background.

PAM matrices extrapolate an evolutionary model from closely related proteins. BLOSUM matrices summarize conserved blocks with sequence clustering to reduce redundancy. Their numbering has different meanings: a higher PAM index describes more evolutionary distance; a lower BLOSUM clustering threshold usually supports more distant comparisons. Standard matrices are loaded from Biopython’s locally installed data.''',
 r'S_{ab}=\log_2\frac{q_{ab}}{p_ap_b}',
'q_ab is the normalized observed pair frequency; p_a and p_b are background residue frequencies. Scores here are bits without integer scaling.',
'If p_A=p_C=0.5 and q_AA=0.4, S_AA=log₂(0.4/0.25)≈0.678 bits.',
'Do not confuse raw count with probability, or PAM numbering with BLOSUM numbering. The lab does not infer a full PAM evolutionary model.',
'O(k²) normalization and score computation for alphabet size k.',
'add pseudocounts to observed pair counts\nnormalize to joint probability q\ncompute expected pairs outer(p,p)\nreturn log2(q / expected)', (ALIGN,('Henikoff & Henikoff: BLOSUM','https://doi.org/10.1073/pnas.89.22.10915'))),
Lesson('search','Sequence search and significance','Pairwise alignment; substitution matrices.',
'Search large collections efficiently while understanding what a score says about chance matches.',
'''FASTA finds short word matches, scores promising diagonals, joins compatible regions, then refines candidates. BLAST finds word hits and extends them; production protein BLAST includes neighborhood words and sophisticated scoring. Our exact-word, ungapped X-drop lab exposes seeding and extension but is deliberately smaller.

PSI-BLAST iterates: search, select credible homologues, align them, build a position-specific scoring matrix (PSSM), and search again. The teaching profile lab uses equal-width ungapped windows and a score inclusion cutoff, not calibrated production E-values. Incorrect inclusions can cause profile drift.

An E-value is an expected count of chance hits at least as strong in the specified search space. A shuffled-sequence null experiment estimates a tail probability for its own null model; it is not automatically a BLAST E-value.''',
 r'E=Kmn\exp(-\lambda S),\qquad p_{\mathrm{emp}}=\frac{1+\#\{S_{null}\ge S_{obs}\}}{B+1}',
'K and λ depend on the scoring/null model; m,n are search lengths; S is raw score; B is shuffle count. The lab requires explicit illustrative K and λ.',
'For K=0.1, m=100, n=1000, λ=0.5, S=20: E≈0.454. Doubling database length doubles E at the same score.',
'A heuristic can miss the best alignment. E is not a probability and may exceed 1. Uncalibrated parameters cannot establish biological significance.',
'The teaching exact-word implementation is O(nm) worst case; production indexes and filters have different practical behavior.',
'find matching words\nextend while score drop ≤ X\nrank hits; compare with exact local DP\nselect profile rows; update PSSM; rescan\nshuffle targets to build a null distribution', (BLAST,PSI)),
Lesson('msa','Multiple sequence alignment','Pairwise alignment; distance and clustering.',
'Align several related sequences to identify conserved positions and build profiles.',
'''The number of DP dimensions grows with the number of sequences, making exact general MSA expensive. Progressive alignment first chooses a guide order/tree, aligns a seed, and adds sequences or profiles. Our lab uses profile-to-sequence DP with summed residue scores and linear gaps.

A guide tree is an alignment heuristic, not automatically a reliable evolutionary tree. Early gap placements influence later steps. Sum-of-pairs scores add the pairwise score of every row pair, treating gap-gap as zero. Conservation and logos summarize columns but depend on the sampled sequences and gap treatment.''',
 r'SP(A)=\sum_{u<v}\sum_c s(A_{uc},A_{vc}),\quad I_c=\log_2 k-H_c',
'A is an alignment; u,v are row indices; c is column; k is alphabet size; H is entropy over nongap residues. Logo heights additionally show nongap occupancy.',
'The column [A,A,C] has three unordered pairs: A/A=2 and two A/C=−1, so its score is 0.',
'An MSA with the highest observed score among two orders is not proven globally optimal. Low diversity can exaggerate conservation.',
'Exact k-sequence DP grows with the product of lengths. Progressive profile-to-sequence alignment is polynomial per addition.',
'compute pairwise distances; derive guide order\nstart with first sequence\nDP-align each next sequence to profile\ninsert gaps into every profile row\ncalculate sum-of-pairs and column statistics', (ALIGN,('Clustal Omega method','https://doi.org/10.1038/msb.2011.75'))),
Lesson('hmm','HMM foundations and inference','Probability; sequences; dynamic programming.',
'Model an observed sequence generated by unobserved states, such as different sequence regions.',
'''A hidden Markov model has an initial state distribution π, transition probabilities A, and emission probabilities B. The first observation is emitted from the initial state, then transitions precede subsequent emissions. Rows are normalized conditional distributions.

Forward sums probabilities of all paths up to each position; Backward summarizes possible suffixes. Their product, normalized, gives state posteriors. Viterbi uses maximum instead of sum to find the most probable complete path. Per-position posterior maxima can describe a different sequence of states. We scale Forward rows and use log-space Viterbi to avoid numerical underflow.''',
 r'\alpha_t(j)=B_j(o_t)\sum_i\alpha_{t-1}(i)A_{ij},\quad \gamma_t(j)\propto\alpha_t(j)\beta_t(j)',
'o_t is observation t; α is forward mass; β is suffix mass; γ is posterior state probability. Displayed α is normalized by a per-time scaling factor.',
'With π=[0.6,0.4] and emission probability of A=[0.8,0.2], the first A has likelihood 0.56 and posterior [0.857,0.143].',
'Forward probabilities are not state posteriors until suffix evidence is included. A zero-likelihood observation sequence has undefined conditional posteriors.',
'O(Ts²) time and O(Ts) storage for T observations and s states.',
'forward: sum predecessors × transition × emission\nbackward: sum successors × transition × next emission\nnormalize alpha × beta\nviterbi: maximize log path score; traceback', (HMM,)),
Lesson('hmm_learning','HMM training and profile alignment','HMM inference; log-odds profiles.',
'Learn hidden state behavior from observations and represent a sequence family with insertions and deletions.',
'''Baum–Welch is expectation maximization. Forward–Backward computes expected state occupancies and transition counts; the M-step normalizes those counts into new probabilities. Exact EM without regularization should not decrease the observed log-likelihood, apart from floating-point error. Different initializations may reach different local optima; state labels are arbitrary.

A profile HMM uses position-specific match emissions, insert states that consume extra query symbols, and silent delete states that skip profile positions. The lab implements a small generative boundary model with explicit match/insert/delete transitions. It is separate from the fixed-state discrete HMM used for Baum–Welch and is not HMMER.''',
 r"A'_{ij}=\frac{\sum_t\xi_t(i,j)}{\sum_t\sum_k\xi_t(i,k)},\quad B'_i(v)=\frac{\sum_{t:o_t=v}\gamma_t(i)}{\sum_t\gamma_t(i)}",
'ξ is expected adjacent-state occupancy; γ is expected state occupancy; v is an emitted symbol. Unvisited rows retain previous parameters.',
'If expected transitions from state 0 are [8,2], the updated row is [0.8,0.2]. A deletion advances the profile index without advancing the query.',
'Do not force delete states to emit a residue. More likelihood on training data does not prove better biology. Zero entries cannot recover without smoothing or changed initialization.',
'EM: O(iterations × Ts²). Small profile Viterbi: O(profile length × query length).',
'repeat: Forward–Backward; expected counts; normalize\nprofile DP:\n M advances profile and query\n I advances query only\n D advances profile only\n traceback best path', (HMM,('HMMER documentation','http://hmmer.org/documentation.html'))),
Lesson('geometry','3D geometry and superposition','Vectors, matrices, protein structures.',
'Remove arbitrary coordinate frames before comparing molecular shapes.',
'''Rigid motion combines rotation with translation. A rotation preserves distances and has orthogonal matrix R with determinant +1. Unit quaternions encode an axis-angle rotation without Euler-angle singularities; q and −q represent the same rotation. Transformation composition depends on order.

Kabsch centers corresponding coordinates, forms their cross-covariance, takes an SVD, corrects an improper reflection, and restores translation. This minimizes squared residual distances over proper rigid rotations. The app uses row vectors: P′=PR+t. Non-collinear points are needed for a stable unique orientation.''',
 r'\mathrm{RMSD}=\sqrt{\frac1N\sum_i\|p_iR+t-q_i\|^2},\quad q=(\cos(\theta/2),\hat u\sin(\theta/2))',
'R is a row-vector rotation; t is translation; p and q are corresponding points; û is a unit axis and θ the rotation angle.',
'Translating all points by [3,0,0] gives raw RMSD 3. After fitting the translation, RMSD is 0, up to rounding.',
'Never mix row-vector and column-vector conventions. Low RMSD depends on correspondence and aligned length. Reflection correction preserves chirality.',
'O(N) covariance accumulation plus a constant-sized 3×3 SVD.',
'center P and Q\nU,S,Vt = SVD(Pc.T @ Qc)\ncorrect determinant sign\nR = U @ correction @ Vt\nt = mean(Q) − mean(P) @ R', (KABSCH,)),
Lesson('distances','Structure from internal distances','3D geometry; eigenvalues and SVD.',
'Reconstruct shape from distances even when no coordinate frame is supplied.',
'''For Euclidean distances, double-centering squared distances gives a centered Gram matrix of dot products. Its leading eigenvectors, scaled by square roots of eigenvalues, give coordinates. For a positive semidefinite Gram matrix this is equivalent to the corresponding SVD construction; eigenvalues additionally reveal non-Euclidean inconsistencies.

Three dimensions retain only the three largest nonnegative eigenvalues. Noise can create negative eigenvalues or require more than three dimensions. Distances do not determine absolute translation, orientation, or handedness: reflected coordinates have exactly the same internal distances.''',
 r'J=I-\frac1N\mathbf1\mathbf1^T,\quad G=-\tfrac12 J D^{\circ2} J,\quad X=V_3\sqrt{\Lambda_3}',
'D contains distances, squared elementwise; J centers; G is the Gram matrix; V₃ and Λ₃ are its leading eigenvectors and nonnegative eigenvalues.',
'For points at (0,0,0),(1,0,0),(0,1,0), distances are 1,1,√2. Reconstruction recovers the triangle with a different orientation.',
'A symmetric nonnegative matrix need not be Euclidean. Clipping negative eigenvalues produces an approximation. Distances alone cannot establish chirality.',
'O(N³) dense eigendecomposition and O(N²) storage.',
'validate symmetric distances and zero diagonal\ndouble-center squared distances\neigendecompose Gram matrix\nretain top three nonnegative components\ncompare reconstructed internal distances', (MDS,('Gower: principal coordinate analysis','https://doi.org/10.1093/biomet/53.3-4.325'))),
Lesson('structure','Structure alignment','Kabsch; sequence alignment.',
'Find which residues should correspond when two protein structures are compared.',
'''Superposition assumes known pairs; structure alignment must infer them. The teaching algorithm initializes ordered pairs, fits a rigid transform, uses distance-based sequence-style DP to update monotone correspondences, and repeats. It is a local heuristic and can become trapped in poor initial correspondences.

RMSD must be reported with pair count: dropping difficult pairs can improve RMSD while reducing coverage. Structure and sequence similarity measure different aspects; geometry can be conserved despite substitutions. Production structural alignment algorithms use more sophisticated objectives and search strategies.''',
 r's(i,j)=3-\|p_iR+t-q_j\|',
'The teaching DP score rewards nearby points in coordinate units (Å for PDB input); gaps penalize skipped residues.',
'Two identical backbones with different amino-acid labels can have fitted RMSD 0 and low sequence identity. Neither result alone proves identical function.',
'Closest neighbors alone can violate residue order. A local heuristic has no global optimality guarantee. RMSD comparisons need consistent units and coverage.',
'O(iterations × nm) distance DP plus rigid fitting.',
'initialize ordered correspondences\nrepeat:\n fit Kabsch on current pairs\n DP-align points using distance scores\n refit on recovered monotone pairs\nreport RMSD and coverage', (KABSCH,('TM-align method','https://doi.org/10.1093/nar/gki524'))),
Lesson('modern','Learning, protein AI and language models','Biology; probability; vectors; training/test split.',
'Connect sequence statistics to learned representations and protein prediction while understanding evidence and limitations.',
'''Secondary-structure classification predicts helix, strand, or coil for each residue. A small neural network can map a one-hot sequence window through a hidden layer to class probabilities. Train/test separation must happen by protein rather than overlapping residue windows. Our synthetic labels reflect designed sequence preferences, not experimental annotations.

Modern structure prediction combines learned sequence/evolutionary representations with geometry. AlphaFold2 uses MSA and pair representations and a structure module; language-model approaches such as ESMFold can derive representations without an input MSA. The lab displays a synthetic prediction/reference pair, confidence-like values, and a PAE-like matrix. These are teaching fixtures, not AlphaFold outputs. pLDDT concerns local confidence; PAE concerns relative positioning confidence, and neither is a measured coordinate error.

Protein language models learn from residue sequences through objectives such as masked-token prediction. Our tiny two-neighbor masked-residue network exposes cross-entropy, hidden embeddings, and residue rankings. It is not a transformer or a pretrained biological foundation model; its synthetic held-out patterns are closely related to training motifs.''',
 r'h=\tanh(XW_1+b_1),\quad p=\mathrm{softmax}(hW_2+b_2),\quad \mathcal L=-\frac1N\sum_i\log p_{i,y_i}',
'X encodes contexts; W,b are trainable parameters; h is the hidden embedding; y is the target class; p is the class probability.',
'If the true residue receives probability 0.25, its loss is −ln(0.25)≈1.386. Confident predictions on synthetic patterns do not establish performance on natural proteins.',
'Split by sequence before extracting windows. Avoid treating confidence as correctness. Dimensionality-reduced embeddings can distort relationships.',
'Dense-network training scales with samples × epochs × layer sizes; production folding systems have much larger costs.',
'split whole sequences into train and test\none-hot encode contexts\nforward pass; compute cross-entropy\nbackpropagate and update weights\nreport held-out performance and inspect embeddings', (AF,ESM,('AlphaFold2 paper','https://doi.org/10.1038/s41586-021-03819-2'))),
]
BY_ID={x.id:x for x in LESSONS}
GLOSSARY={
'homology':'Shared evolutionary ancestry; it is a relationship, not a percentage.',
'identity':'Fraction of aligned positions containing the same residue; specify gap treatment.',
'codon':'Three nucleotides read together during translation.',
'RMSD':'Root mean squared distance between corresponding coordinates after a stated fit.',
'PSSM':'Position-specific scoring matrix describing residue preferences in a family.',
'posterior':'A conditional distribution after incorporating observed evidence.',
'quaternion':'Four-component representation of a 3D rotation; unit norm is required.',
'cross-entropy':'Negative log probability of the true class, averaged over examples.',
'pLDDT':'Predicted local confidence; not a measured coordinate error.',
'PAE':'Predicted aligned error indicating confidence in relative positioning.',
}
