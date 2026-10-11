"""Run with: streamlit run app.py"""
import streamlit as st
from bioapp.content import LESSONS
from bioapp import views

st.set_page_config(page_title='Computational Biology Lab',page_icon='🧬',layout='wide')
st.markdown('''<style>
abbr[title] {text-decoration: underline dotted; cursor: help; font-weight: 600;}
.block-container {max-width: 1400px; padding-top: 2rem;}
</style>''',unsafe_allow_html=True)
views.init()
with st.sidebar:
    st.title('🧬 Bioinformatics 2026')
    options=['Course overview']+[l.title for l in LESSONS]+['Practice exam','Assignments','Progress and review']
    page=st.selectbox('Navigate',options,key='navigation')
    st.caption('All topics are unlocked. Follow the course order or explore freely.')
    views.progress_panel()
    if st.button('Clear experiment cache'): st.cache_data.clear(); st.success('Cached model training cleared.')
if page=='Course overview': views.home()
elif page=='Practice exam': views.exam()
elif page=='Assignments': views.assignments()
elif page=='Progress and review': views.review()
else: views.topic(next(l for l in LESSONS if l.title==page))
