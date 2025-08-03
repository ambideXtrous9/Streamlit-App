import streamlit as st 
import pandas as pd
import numpy as np
import requests
import torch 
import os



os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# UI configurations
st.set_page_config(page_title="ambideXtrous",
                   page_icon=":bridge_at_night:",
                   layout="centered")

                   
from sidebar import SideBar
from navigate import navigator


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
# torch.classes.__path__ = []

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    

    # If a username is present in session_state, assume logged in for persistence across refreshes
    # Check URL query parameters for persistence
    if "logged_in_user" in st.query_params and st.query_params["logged_in_user"]:
        st.session_state['username'] = st.query_params["logged_in_user"]
        st.session_state['logged_in'] = True
    elif st.session_state['username']:
        st.session_state['logged_in'] = True

    SideBar()
    navigator()

if __name__ == '__main__':
    main()


