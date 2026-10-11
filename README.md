# Computational Biology Learning Lab

A standalone, English, offline Streamlit study companion for the supplied 2026
Introduction to Computational Biology syllabus. It provides twelve complete topic
modules with lessons, equations, worked examples, pseudocode, editable experiments,
36 graded questions, a configurable practice exam, and three original assignments.
It is independent of the MLcourse app and does not require slides.

## Run

Use Python 3.11 or newer:

```bash
cd "/Users/edeneldar/Intro to Bioinformatics 2026"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address Streamlit prints (normally http://localhost:8501).
Installation requires package downloads once. Lessons, computations, figures,
fixtures, standard substitution matrices, and quizzes then run offline. Source
links are optional online reading. No API keys, external CDNs, GPUs, or model
weights are used. The project-local `.venv` is not part of the source deliverable.

## Study workflow

1. Pick a topic from the sidebar or course overview.
2. Read **Learn**, including prerequisites and the worked example.
3. Predict the answer, edit **Experiment** inputs, and press **Run experiment**.
   Results persist while stepping through a trace. Changed inputs only apply
   after another run; results explicitly identify this behavior.
4. Use **Practice** to submit conceptual, numeric, and tracing answers. Hints and
   solutions become available after submission. Numeric tolerance is 0.01.
5. Inspect source code and sources in **Reference**, then mark the lesson complete.
6. Use **Practice exam**, **Assignments**, and **Progress and review** for revision.

Export progress from the sidebar before ending a session. Import and explicitly
restore that JSON to continue later. Progress is not silently written to disk and
there is no login or shared student database. Imports replace the current progress
only after validation. Only trusted progress files should be used.

The exam records a deadline and checks it at submission; its remaining-time display
updates on interaction or the Refresh button, rather than an automatic countdown.
Late submissions are visibly marked and remain useful for self-study.

## Coverage and implementation

- `bioapp/content.py`: twelve syllabus-led lessons, references, and terminology.
- `bioapp/sequences.py`: affine global/local alignment, rolling rows, Hirschberg,
  log-odds scores, profile MSA, exact-word X-drop search and profile scanning.
- `bioapp/hmm.py`: scaled Forward–Backward, log Viterbi, Baum–Welch and a small
  generative profile-HMM with silent deletions.
- `bioapp/geometry.py`: quaternions, proper Kabsch fits, distance reconstruction,
  ordered distance-DP structural correspondence and PDB C-alpha parsing.
- `bioapp/models.py`: readable NumPy MLPs for synthetic secondary structure and
  masked-residue prediction. Protein splits precede window extraction.
- `bioapp/labs.py`, `bioapp/views.py`: interactive labs and learning interface.
- `bioapp/assessment.py`: questions, grading, assignments and progress schema v1.

Sequence inputs are limited per lab (generally 60–200 residues); alignments have
an algorithm-level 400-symbol ceiling. PDBs are limited to 500 KB and 200 C-alpha
points. FASTA uploads are limited to 100 KB. The first PDB model is used, with
blank/A alternate locations and duplicate residues removed. Unknown alphabets,
nonfinite matrices, inconsistent probabilities and degenerate fits are rejected.

### Interpretation

FASTA/BLAST/PSI-BLAST labs are labeled simplified teaching algorithms, not wrappers
around production tools. E-values use user-specified illustrative parameters and
are not calibrated. The shuffled-target experiment has its own composition-based
null. Structure alignment is a local heuristic. All bundled fixtures are original
synthetic examples; confidence-like values are not outputs from AlphaFold. Tiny
networks are not biological predictors or pretrained language models. See
`data/README.md` for provenance and the lessons for primary references.

The syllabus mentions unspecified optional topics: these remain an extension area.
To add future slides, place them in a resources directory, add local download links,
and reconcile the relevant lesson text and questions with the instructor's scope.

## Validation

```bash
python -m pip install pytest
python -m pytest -q
```

Tests compare affine alignment with Biopython, compare HMM inference with exhaustive
path sums, check EM likelihoods, rigid geometry and internal-distance recovery,
validate malformed inputs and progress, and exercise each Streamlit page, lab,
practice answer, and exam flow with AppTest. No network is needed to run the tests.

The syllabus PDF remains user-supplied and is excluded from the MIT license covering
this project's code, original explanations, questions, and synthetic fixtures.
