import streamlit as st 
import pandas as pd
import numpy as np
import requests

from sidebar import SideBar
from navigate import navigator

# UI configurations
st.set_page_config(page_title="ambideXtrous",
                   page_icon=":bridge_at_night:",
                   layout="centered")

SideBar()

navigator()


