"""Course navigation, learning, assessment, and portable personal progress."""
import html
import inspect
import json
import random
import re
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from .content import LESSONS, BY_ID, GLOSSARY
from .assessment import QUESTIONS, QUESTION_BY_ID, ASSIGNMENTS, record_attempt, load_progress, fresh_progress
from .labs import LABS
from . import sequences, hmm, geometry, models
from .datasets import backbone, pdb_text


def init():
    if 'progress' not in st.session_state: st.session_state['progress']=fresh_progress()


def open_lesson(title):
    st.session_state['navigation']=title


def progress_panel():
    p=st.session_state['progress']
    with st.sidebar.expander('Save / restore progress'):
        st.caption('Progress stays in this browser session. Export a file to retain it after a restart; no accounts are needed.')
        st.download_button('Export progress',json.dumps(p,indent=2),'bioinformatics_progress.json','application/json')
        upload=st.file_uploader('Import progress JSON',type=['json'],key='progress_upload')
        if st.button('Restore imported progress'):
            try:
                if upload is None: raise ValueError('Choose a progress JSON file first.')
                st.session_state['progress']=load_progress(upload.getvalue())
                for key in list(st.session_state):
                    if key.startswith('feedback_') or key=='exam': del st.session_state[key]
                st.success('Progress restored.')
            except ValueError as exc: st.error(str(exc))
        if st.checkbox('Confirm progress reset',key='confirm_reset') and st.button('Reset progress'):
            st.session_state['progress']=fresh_progress()
            for key in list(st.session_state):
                if key.startswith('feedback_') or key=='exam': del st.session_state[key]
            st.rerun()


def topic_stats(topic):
    attempts=[a for a in st.session_state['progress']['attempts'] if QUESTION_BY_ID[a['question_id']].topic==topic]
    latest={a['question_id']:a for a in attempts}
    accuracy=sum(a['correct'] for a in latest.values())/len(latest) if latest else None
    return len(attempts),accuracy


def home():
    st.title('🧬 Introduction to Computational Biology')
    st.markdown('**Learn the biology. Trace the algorithms. Experiment, practice, and review.**')
    st.caption('2026 syllabus · Prof. Rachel Kolodny · Independent study companion; not official lecture material.')
    p=st.session_state['progress']; cols=st.columns(3)
    cols[0].metric('Lessons completed',f"{len(p['completed'])} / {len(LESSONS)}")
    cols[1].metric('Practice attempts',len(p['attempts']))
    cols[2].metric('Questions available',len(QUESTIONS))
    st.progress(len(p['completed'])/len(LESSONS))
    st.markdown('### Course map')
    groups=[('Biology',['biology']),('Sequences',['pairwise','linear','matrices','search','msa']),('Hidden states',['hmm','hmm_learning']),('3D structures',['geometry','distances','structure']),('Protein AI',['modern'])]
    fig=go.Figure(); xpos=[]; ypos=[]; names=[]; colors=[]
    for x,(group,ids) in enumerate(groups):
        for y,id_ in enumerate(ids):
            xpos.append(x); ypos.append(-y); names.append(BY_ID[id_].title); colors.append('#10b981' if id_ in p['completed'] else '#38bdf8')
            if y: fig.add_trace(go.Scatter(x=[x,x],y=[-y+1,-y],mode='lines',line=dict(color='#cbd5e1'),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=list(range(5)),y=[.7]*5,mode='lines',line=dict(color='#64748b'),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xpos,y=ypos,mode='markers+text',text=names,textposition='bottom center',marker=dict(size=18,color=colors),hovertext=names,hoverinfo='text',showlegend=False))
    fig.update_layout(height=450,xaxis=dict(tickvals=list(range(5)),ticktext=[g for g,_ in groups],range=[-.5,4.5]),yaxis=dict(visible=False,range=[-4.8,1]),margin=dict(l=10,r=10,t=10,b=10),plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig,width='stretch'); st.caption('Green indicates marked-complete lessons. Use the navigation menu or lesson buttons below to open a topic.')
    for lesson in LESSONS:
        count,acc=topic_stats(lesson.id)
        c1,c2,c3=st.columns([4,2,1])
        c1.write(f'**{lesson.title}**'); c2.caption('Not practiced' if acc is None else f'{count} attempts · latest accuracy {acc:.0%}')
        c3.button('Open',key='open_'+lesson.id,on_click=open_lesson,args=(lesson.title,))
    st.markdown('### Suggested learning path')
    st.write('Follow the numbered topics in order. Start each with Learn, predict an output before running Experiment, then answer Practice without looking at the solution.')
    st.info('The syllabus specifies an exam worth 90% and three assignments worth 10%. The mock exam and assignments here are original study exercises and do not predict the instructor’s grading or exam questions.')
    st.markdown('### Extensions and future slides')
    st.write('The syllabus leaves “other topics” unspecified. Additional lecture material can be attached and reconciled with these lessons later; no extra topic is treated as required.')
    syllabus=Path(__file__).resolve().parents[1]/'Syllabus.pdf'
    if syllabus.exists(): st.download_button('Download supplied syllabus',syllabus.read_bytes(),'Syllabus.pdf','application/pdf')
    st.download_button('Download synthetic backbone PDB',pdb_text(backbone()),'synthetic_helix.pdb','chemical/x-pdb')


def learn(lesson):
    st.caption('Prerequisites: '+lesson.prerequisites)
    st.markdown('### Why this matters'); st.write(lesson.motivation)
    st.markdown('### Core ideas')
    pattern=re.compile(r'\b('+ '|'.join(re.escape(x) for x in sorted(GLOSSARY,key=len,reverse=True)) +r')\b',re.IGNORECASE)
    lookup={k.lower():v for k,v in GLOSSARY.items()}
    notes=pattern.sub(lambda m:f'<abbr title="{html.escape(lookup[m.group().lower()],quote=True)}">{html.escape(m.group())}</abbr>',lesson.notes)
    st.markdown(notes,unsafe_allow_html=True)
    with st.expander('Math and algorithm prerequisites: quick refresher'):
        st.markdown('''**Probability:** conditional probabilities describe a symbol given a state. A distribution sums to one. Multiply probabilities along one complete path; add probabilities over mutually exclusive alternative paths. Bayes’ rule converts a joint probability into a posterior by dividing by the evidence.

**Logs:** log(ab)=log(a)+log(b). A log-odds score is positive when observed probability exceeds its background probability. Natural logs are used for likelihood and neural-network loss; base-2 logs measure bits.

**Dynamic programming:** define what each cell means, initialize empty-prefix cases, then reuse already-solved smaller problems. A predecessor pointer records a maximizing choice; tracing pointers recovers a solution. Ties can yield several optimal solutions.

**Vectors and matrices:** a 3D point has three coordinates. Euclidean distance is the square root of the sum of squared coordinate differences. Matrix multiplication composes linear transformations; transpose exchanges rows and columns. An orthogonal matrix preserves lengths. SVD factors a matrix into two orthogonal factors and nonnegative singular values; an eigendecomposition finds directions scaled by a square symmetric matrix.

**Cost and learning:** O(nm) means work scales with the product of two input lengths. A gradient describes how a small parameter change affects loss; gradient descent subtracts learning-rate times gradient. Training fits parameters, validation helps choose settings, and a held-out test checks performance on unseen examples.''')
    st.latex(lesson.formula); st.write(lesson.symbols)
    st.markdown('### Worked example'); st.write(lesson.worked)
    st.markdown('### Algorithm'); st.code(lesson.pseudocode,language='text')
    st.write('**Computational cost:** '+lesson.complexity)
    st.markdown('### Common mistakes'); st.write(lesson.pitfalls)
    completed=lesson.id in st.session_state['progress']['completed']
    if st.button('Mark lesson complete' if not completed else 'Mark as needing review',key='complete_'+lesson.id):
        if completed: st.session_state['progress']['completed'].remove(lesson.id)
        else: st.session_state['progress']['completed'].append(lesson.id)
        st.rerun()


def answer_widget(q,prefix):
    if q.options: return st.radio(q.prompt,q.options,index=None,key=prefix+'_'+q.id)
    return st.text_input(q.prompt,key=prefix+'_'+q.id,placeholder='Enter a number')


def practice(lesson):
    st.write('Submit an answer to reveal feedback and a hint. Numeric answers use an absolute tolerance of 0.01.')
    for q in (x for x in QUESTIONS if x.topic==lesson.id):
        with st.container(border=True):
            with st.form('practice_'+q.id):
                value=answer_widget(q,'practice')
                submitted=st.form_submit_button('Check answer')
            if submitted:
                if value is None or value=='': st.warning('Choose or enter an answer before submitting.')
                else:
                    ok=record_attempt(st.session_state['progress'],q,value)
                    st.session_state['feedback_'+q.id]=ok
            if 'feedback_'+q.id in st.session_state:
                if st.session_state['feedback_'+q.id]: st.success('Correct.')
                else: st.error('Not yet. Use the hint and try again.')
                with st.expander('Hint'): st.write(q.hint)
                with st.expander('Answer and explanation'): st.write(f'**Answer:** {q.answer}'); st.write(q.explanation)


REFERENCE_FUNCTIONS={
'biology':[], 'pairwise':[sequences.align], 'linear':[sequences.rolling_row,sequences.hirschberg],
'matrices':[sequences.log_odds], 'search':[sequences.seed_search,sequences.pssm_from_rows,sequences.profile_scan],
'msa':[sequences.progressive], 'hmm':[hmm.infer], 'hmm_learning':[hmm.baum_welch,hmm.profile_viterbi],
'geometry':[geometry.quaternion_rotation,geometry.kabsch], 'distances':[geometry.reconstruct],
'structure':[geometry.structure_align], 'modern':[models.train_mlp,models.predict,models.windows]}


def reference(lesson):
    st.markdown('### Sources')
    for title,url in lesson.sources: st.markdown(f'- [{title}]({url})')
    st.caption('The lessons are original syllabus-led explanations. Source links are optional reading; experiments work offline.')
    st.markdown('### Inspect the implementation')
    if not REFERENCE_FUNCTIONS[lesson.id]:
        from Bio.Seq import Seq
        st.code("from Bio.Seq import Seq\ndna = Seq('ATGGAATAA')\nprint(dna.reverse_complement())\nprint(dna.transcribe())\nprint(dna.translate())",language='python')
    for fn in REFERENCE_FUNCTIONS[lesson.id]:
        with st.expander(fn.__name__): st.code(inspect.getsource(fn),language='python')
    st.markdown('### Terminology')
    for word,definition in GLOSSARY.items():
        st.markdown(f'<abbr title="{html.escape(definition,quote=True)}">{html.escape(word)}</abbr> — {html.escape(definition)}',unsafe_allow_html=True)


def topic(lesson):
    st.title(lesson.title)
    section=st.radio('Study mode',['Learn','Experiment','Practice','Reference'],horizontal=True,key='mode_'+lesson.id)
    if section=='Learn': learn(lesson)
    elif section=='Experiment': LABS[lesson.id]()
    elif section=='Practice': practice(lesson)
    else: reference(lesson)


def exam():
    st.title('Practice exam')
    st.caption('Original mixed questions. No official exam weighting is inferred. Answers remain hidden until submission.')
    if 'exam' not in st.session_state:
        topics=st.multiselect('Topics',[l.title for l in LESSONS],default=[l.title for l in LESSONS])
        number=st.slider('Requested number of questions',3,len(QUESTIONS),12)
        minutes=st.slider('Time limit (minutes; 0 = untimed)',0,60,20)
        seed=st.number_input('Exam random seed',0,999,7,key='exam_seed')
        if st.button('Start exam',type='primary'):
            ids={l.id for l in LESSONS if l.title in topics}; pool=[q.id for q in QUESTIONS if q.topic in ids]
            if not pool: st.error('Select at least one topic.')
            else:
                st.session_state['exam']={'ids':random.Random(seed).sample(pool,min(number,len(pool))),'started':time.time(),'deadline':time.time()+60*minutes if minutes else None,'submitted':False}
                st.rerun()
        return
    e=st.session_state['exam']
    if not e['submitted']:
        remaining=None if e['deadline'] is None else max(0,e['deadline']-time.time())
        if remaining is not None: st.info(f'Time remaining at last interaction: {int(remaining//60)}m {int(remaining%60)}s. The deadline is checked at submission.')
        if st.button('Refresh remaining time'): st.rerun()
        with st.form('exam_answers'):
            values={id_:answer_widget(QUESTION_BY_ID[id_],'exam_'+str(e['started'])) for id_ in e['ids']}
            submit=st.form_submit_button('Submit exam')
        if submit:
            late=e['deadline'] is not None and time.time()>e['deadline']
            if not late and any(v is None or v=='' for v in values.values()): st.warning('Answer every question before submitting, or wait for the deadline to submit unfinished work.')
            else:
                e['answers']={k:(v if v is not None else '') for k,v in values.items()}
                e['scores']={k:record_attempt(st.session_state['progress'],QUESTION_BY_ID[k],v,'exam') for k,v in e['answers'].items()}; e['submitted']=True; e['late']=late; st.rerun()
    else:
        st.metric('Score',f"{sum(e['scores'].values())} / {len(e['ids'])}")
        if e['late']: st.warning('Submitted after the deadline; this practice score is marked late.')
        for id_ in e['ids']:
            q=QUESTION_BY_ID[id_]
            with st.expander(('✓ ' if e['scores'][id_] else 'Review: ')+q.prompt):
                st.write('Your answer:',e['answers'][id_]); st.write('Correct answer:',q.answer); st.write(q.explanation)
    if st.button('Discard exam / configure a new one'): del st.session_state['exam']; st.rerun()


def assignments():
    st.title('Three practice assignments')
    st.caption('Original self-study assignments; these are not the instructor’s assignments. Use the labs to verify your work.')
    for i,(title,tasks,rubric) in enumerate(ASSIGNMENTS,1):
        with st.container(border=True):
            st.subheader(f'{i}. {title}'); st.markdown(tasks)
            with st.expander('Solution checklist'): st.write(rubric)
            st.download_button('Download assignment',f'# Assignment {i}: {title}\n\n{tasks}\n\n## Solution checklist\n{rubric}\n',f'assignment_{i}.md','text/markdown',key='assignment_download_'+str(i))


def review():
    st.title('Progress and review')
    rows=[]
    for lesson in LESSONS:
        n,acc=topic_stats(lesson.id); rows.append({'topic':lesson.title,'lesson complete':lesson.id in st.session_state['progress']['completed'],'attempts':n,'latest question accuracy':acc,'needs review':acc is not None and acc<.8})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.caption('Accuracy uses the most recent attempt on each answered question. Unanswered questions do not count as correct or incorrect; lesson completion is self-reported.')
    history=st.session_state['progress']['attempts']
    if history: st.dataframe(pd.DataFrame(history),hide_index=True)
