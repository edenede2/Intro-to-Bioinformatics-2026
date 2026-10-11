from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from bioapp.content import LESSONS
from bioapp.assessment import QUESTIONS, fresh_progress, load_progress

APP=Path(__file__).resolve().parents[1]/'app.py'


def app():
    return AppTest.from_file(str(APP),default_timeout=30).run()


def navigate(at,page):
    at.selectbox(key='navigation').select(page).run()
    assert not at.exception
    return at


@pytest.mark.parametrize('lesson',LESSONS,ids=lambda l:l.id)
def test_every_lesson_lab_practice_reference(lesson):
    at=navigate(app(),lesson.title)
    assert not at.exception and at.latex
    at.radio(key='mode_'+lesson.id).set_value('Experiment').run()
    assert not at.exception
    next(b for b in at.button if b.label=='Run experiment').click().run()
    assert not at.exception
    assert not at.error
    assert 'lab_'+({'hmm_learning':'em','modern':'secondary'}.get(lesson.id,lesson.id)) in at.session_state
    at.radio(key='mode_'+lesson.id).set_value('Practice').run()
    q=next(q for q in QUESTIONS if q.topic==lesson.id)
    if q.options: at.radio(key='practice_'+q.id).set_value(q.answer)
    else: at.text_input(key='practice_'+q.id).set_value(str(q.answer))
    next(b for b in at.button if b.label=='Check answer').click().run()
    assert not at.exception and at.success
    assert at.session_state['progress']['attempts'][-1]['correct']
    exported=json.dumps(at.session_state['progress'])
    assert load_progress(exported)==at.session_state['progress']
    at.radio(key='mode_'+lesson.id).set_value('Reference').run()
    assert not at.exception and at.code


@pytest.mark.parametrize('page',['Assignments','Progress and review','Practice exam'])
def test_other_pages(page):
    assert not navigate(app(),page).exception


def test_complete_and_restore_progress():
    at=navigate(app(),LESSONS[0].title)
    at.button(key='complete_biology').click().run()
    assert 'biology' in at.session_state['progress']['completed']
    saved=load_progress(json.dumps(at.session_state['progress']))
    restarted=app(); restarted.session_state['progress']=saved
    navigate(restarted,'Progress and review')
    assert restarted.session_state['progress']['completed']==['biology']


def test_mixed_exam():
    at=navigate(app(),'Practice exam')
    next(b for b in at.button if b.label=='Start exam').click().run()
    assert not at.exception
    e=at.session_state['exam']
    for id_ in e['ids']:
        q=next(q for q in QUESTIONS if q.id==id_); key='exam_'+str(e['started'])+'_'+id_
        if q.options: at.radio(key=key).set_value(q.answer)
        else: at.text_input(key=key).set_value(str(q.answer))
    next(b for b in at.button if b.label=='Submit exam').click().run()
    assert not at.exception
    assert all(at.session_state['exam']['scores'].values())
    assert len(at.session_state['progress']['attempts'])==12


@pytest.mark.parametrize('page,label,value,result_key',[
('HMM training and profile alignment','Lab','Profile-HMM alignment','profile'),
('Learning, protein AI and language models','Experiment','Structure prediction confidence','prediction'),
('Learning, protein AI and language models','Experiment','Masked residues and embeddings','masked'),
])
def test_alternate_labs(page,label,value,result_key):
    at=navigate(app(),page); lesson=next(l for l in LESSONS if l.title==page)
    at.radio(key='mode_'+lesson.id).set_value('Experiment').run()
    next(r for r in at.radio if r.label==label).set_value(value).run()
    next(b for b in at.button if b.label=='Run experiment').click().run()
    assert not at.exception and not at.error
    assert 'lab_'+result_key in at.session_state


def test_invalid_input_is_actionable():
    at=navigate(app(),LESSONS[0].title); at.radio(key='mode_biology').set_value('Experiment').run()
    at.text_area(key='bio_seq').set_value('NONSENSE')
    at.button(key='run_biology').click().run()
    assert not at.exception and at.error


def test_overview_open_button():
    at=app(); at.button(key='open_pairwise').click().run()
    assert not at.exception
    assert at.selectbox(key='navigation').value=='Pairwise sequence alignment'


@pytest.mark.parametrize('method',['FASTA-like diagonal filtering','PSI-like profile iterations'])
def test_search_alternatives(method):
    at=navigate(app(),'Sequence search and significance'); at.radio(key='mode_search').set_value('Experiment').run()
    next(s for s in at.selectbox if s.label=='Teaching search mode').set_value(method)
    at.button(key='run_search').click().run()
    assert not at.exception and not at.error
    if method.startswith('PSI'): assert at.session_state['lab_search'][2]


def test_restore_control(monkeypatch):
    import io
    import streamlit as st
    p=fresh_progress(); p['completed']=['biology','hmm']; payload=io.BytesIO(json.dumps(p).encode())
    original=st.file_uploader
    def upload(label,*args,**kwargs):
        if kwargs.get('key')=='progress_upload': return payload
        return original(label,*args,**kwargs)
    monkeypatch.setattr(st,'file_uploader',upload)
    at=app(); next(b for b in at.button if b.label=='Restore imported progress').click().run()
    assert not at.exception and at.session_state['progress']==p


def test_malformed_hmm_and_progress_control(monkeypatch):
    at=navigate(app(),'HMM foundations and inference'); at.radio(key='mode_hmm').set_value('Experiment').run()
    at.text_input(key='hmm_pi').set_value('0.5'); at.button(key='run_hmm').click().run()
    assert not at.exception and at.error


def test_timed_exam_late_submission():
    at=navigate(app(),'Practice exam'); next(b for b in at.button if b.label=='Start exam').click().run()
    e=at.session_state['exam']; e['deadline']=0; at.session_state['exam']=e
    next(b for b in at.button if b.label=='Submit exam').click().run()
    assert not at.exception and at.session_state['exam']['submitted'] and at.session_state['exam']['late']
    assert not any(at.session_state['exam']['scores'].values())


def test_core_flow_without_network(monkeypatch):
    import socket
    def reject(*args,**kwargs): raise AssertionError('Offline app tried to open a network connection')
    monkeypatch.setattr(socket,'create_connection',reject)
    monkeypatch.setattr(socket.socket,'connect',reject)
    at=navigate(app(),'Learning, protein AI and language models')
    at.radio(key='mode_modern').set_value('Experiment').run()
    at.button(key='run_secondary').click().run()
    assert not at.exception and not at.error
