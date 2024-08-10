import streamlit as st 
from util import GitHubStats, HomePage, Social, YoloforLogo, NewsQA


def navigator():

    # Get the current page from the query parameters
    
    page = "page"
    
    if page in st.query_params.keys():
        page = st.query_params["page"]
    else: page = "home"

    # Display content based on the selected menu item
    if page == "home":
        HomePage()
        

    elif page == "Home":
        HomePage()

    # Page content based on the selected menu item
    elif page == "yolologo":
        st.title(":rainbow[YOLO for Logo!] :sunglasses:")
        YoloforLogo()

    elif page == "github_stats":
        GitHubStats()
        

    elif page == "Social":
        Social(sidebarPos=False,heading="Social")


    elif page == "newsqa":
        st.title(":rainbow[⚙️ News QA System using LLM] :sunglasses:")
        NewsQA()
        

    else:
        st.subheader("Home Page")
        st.write("Welcome to the Streamlit Portfolio. Use the sidebar to navigate.")
