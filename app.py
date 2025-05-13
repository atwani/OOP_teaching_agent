"""
Entrypoint for streamlit app.
Runs top to bottom every time the user interacts with the app (other than imports and cached functions).
"""

# Library imports
import traceback
import copy

import streamlit as st


from utils.page_components import (
    add_common_page_elements,
)

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

displaytext= ('''## About somaGPT ''')

st.markdown(displaytext)

displaytext= (
    '''SomaGPT is a platform for learning programming in a socratic manner. \n\n'''
    '''It uses foundational LLMS such as GPT and Gemini. '''
    '''Instead of the LLM giving you answers it instructs you in an adaptive way by leveraging on your prior knowledge \n\n'''

 )

st.markdown(displaytext)