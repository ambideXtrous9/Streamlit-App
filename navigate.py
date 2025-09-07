import streamlit as st 
from util import GitHubStats, HomePage, Social, YoloforLogo
from Clustering.cluster_util import showData,Cluster
from ImageClassifier.classifier import model_card
from StockScreener.screener import StockScan
from HarryAgent.chatbot import ChatBot
from AirbnbAgent.tourAgent import tourChat
from login import login_page

def navigator():

    # Get the current page from the query parameters
    
    if 'page' in st.session_state:
        page = st.session_state['page']
    elif "page" in st.query_params.keys():
        page = st.query_params["page"]
    else:
        page = "home"

    # Display content based on the selected menu item
    if page == "home":
        HomePage()
        

    elif page == "Home":
        HomePage()

    # Page content based on the selected menu item
    elif page == "signup":
        st.session_state['show_signup'] = True
        st.session_state['show_login'] = False
        login_page()
        
    elif page == "yolologo":
        st.title("🚀:blue[YOLO for Logo!] :sunglasses:")
        YoloforLogo()
        
    elif page == "newsqa":
        if st.session_state.get('logged_in'):
            st.title("🪄:blue[Harry Potter X  Mythology]")
            st.write("Ask anything about Harry Potter and Indian Mythology")
            ChatBot()
        else:
            st.session_state['redirect_after_login'] = "newsqa"
            login_page()
        
    elif page == "image_classifer":
        st.title("🚀:blue[Image Classification ] :sunglasses:")
        model_card()
        
    elif page == "clusterplay":
        st.title("🐙:blue[Play with Clusters] :sunglasses:")
        Cluster()
        
    elif page == "stockscreener":
        if st.session_state.get('logged_in'):
            st.title("🚀:blue[Stock Screener]")
            st.write("This is a Stock Screener that can help you find stocks that are breaking out with volume.")
            st.title("Range Breakout with Volume")
            StockScan()
        else:
            login_page()

    elif page == "tourAgent":
        if st.session_state.get('logged_in'):
            st.title("🏡:blue[MCP Powered Tour Agent]")
            st.write("This is a Weather and MCP powered Airbnb tour agent that can help you plan your next vacation.")
            tourChat()
        else:
            login_page()
        
        
    elif page == "Social":
        Social(sidebarPos=False,heading="Social")

        

    elif page == "login":
        login_page()
